import torch
import torch.nn.functional as F
from torch import Tensor
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from typing import Callable, Tuple
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, cohen_kappa_score, f1_score
from copy import deepcopy
import random
import os
from torchvision.transforms import v2


from models.models import CVAE, CVAEWithKLAnnealing

##### CVAE TRAINING #####

def cvae_fit(model, opt, train_dl, test_dl=None, epochs=20, device="cuda", beta=1, 
             cutmix=False, vae=False, class_weighted=False, spc=None):
    model.to(device)
    
    if cutmix:
        apply_cutmix = v2.CutMix(num_classes=model.n_classes, alpha=0.9)
        
    if class_weighted:
        assert spc is not None, 'spc must be provided for class weighted loss'
        # class_weights = torch.tensor([1.0 / count for count in spc], device=device)
        # class_weights = class_weights / class_weights.max()
        class_weights = (spc.log().max() - spc.log() + 1).to(device)
        # class_weights = (spc.max() - spc + 1).to(device)

    total_loss_epoch_train = []
    rec_loss_epoch_train = []
    kl_loss_epoch_train = []
    
    total_loss_epoch_test = []
    rec_loss_epoch_test = []
    kl_loss_epoch_test = []
    
    for epoch in tqdm(range(epochs)):
        model.train()
        for x, y in train_dl:
            x = x.to(device)
            y_oh = F.one_hot(y, num_classes=model.n_classes).to(device)
            
            if cutmix:
                x, y_oh = apply_cutmix(x.unsqueeze(dim=1).unsqueeze(dim=1), y)
                x = x.squeeze().to(device)
                y_oh = y_oh.to(device)
            
            if vae:
                preds = model(x)
            else:
                preds = model(x, y_oh)
            
            rec_loss = ((x - preds)**2).sum(dim=1)
            if class_weighted:
                rec_loss = (rec_loss * class_weights[y]).mean()
            else:
                rec_loss = rec_loss.mean()
                
            kl_loss = model.encoder.kl / x.size(0)
            if isinstance(model, CVAEWithKLAnnealing):
                loss = rec_loss + kl_loss * model.kl_weight
            else:
                loss = rec_loss + kl_loss * beta
                
            loss.backward()
            opt.step()
            opt.zero_grad()
            
        total_loss_epoch_train.append(loss)
        rec_loss_epoch_train.append(rec_loss)
        kl_loss_epoch_train.append(kl_loss)
        
        if isinstance(model, CVAEWithKLAnnealing):
            model.update_kl_weight(epoch)
            
        if test_dl is not None:
            model.eval()
            with torch.no_grad():
                for x, y in test_dl:
                    x = x.to(device)
                    y_oh = F.one_hot(y, num_classes=model.n_classes).to(device)
                    if vae:
                        preds = model(x)
                    else:
                        preds = model(x, y_oh)
                    
                    rec_loss = ((x - preds)**2).sum() / x.size(0)
                    kl_loss = model.encoder.kl / x.size(0)
                    if isinstance(model, CVAEWithKLAnnealing):
                        loss = rec_loss + kl_loss * model.kl_weight
                    else:
                        loss = rec_loss + kl_loss * beta
                total_loss_epoch_test.append(loss)
                rec_loss_epoch_test.append(rec_loss)
                kl_loss_epoch_test.append(kl_loss)
    
    return {
        'train': {
            'total': total_loss_epoch_train,
            'rec': rec_loss_epoch_train,
            'kl': kl_loss_epoch_train
        },
        'test': {
            'total': total_loss_epoch_test,
            'rec': rec_loss_epoch_test,
            'kl': kl_loss_epoch_test
        }
    }

##### N_Layer_Dense_Classifier TRAINING #####    

# remix implementation
def remix_data(x, y, spc, alpha=1.0, kappa=3, tau=0.5):
    
    # one-hot encode y if not alread
    if y.ndim == 1:
        y = F.one_hot(y, len(spc))

    lam = torch.distributions.beta.Beta(alpha, alpha).sample((y.shape[0],)).to(x.device)

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    # mix x
    mixed_x = lam.unsqueeze(dim=1) * x + (1 - lam.unsqueeze(dim=1)) * x[index, :]
    
    # get ratios
    y_a, y_b = y, y[index]

    n_a = torch.tensor([spc[torch.argmax(i)] for i in y_a])
    n_b = torch.tensor([spc[torch.argmax(i)] for i in y_b])
    ratios = (n_a / n_b).to(x.device)
    
    # determine lamdas
    lam_y = lam.clone()
    
    condition_zero = (ratios >= kappa) & (lam < tau)
    lam_y[torch.nonzero(condition_zero).squeeze()] = 0
    
    condition_one = (ratios <= 1/kappa) & ((1-lam) < tau)
    lam_y[torch.nonzero(condition_one).squeeze()] = 1
    
    # mix y
    mixed_y = lam_y.unsqueeze(dim=1) * y_a + (1 - lam_y.unsqueeze(dim=1)) * y_b
    return mixed_x.to(x.device), mixed_y.to(x.device)


def accuracy(out, y):
    preds = torch.argmax(out, dim=1)
    if y.ndim == 2:
        y = torch.argmax(y, dim=1)
    return (preds == y).float().mean()


def average(metric: tuple, nums: tuple):
    return sum(torch.tensor(metric) * torch.tensor(nums)) / sum(torch.tensor(nums))


def loss_batch(X, y, model, loss_fn, opt=None, acc=False, remix=False, spc=None):
    if remix:
        X, y = remix_data(X, y, spc)
    
    preds = model(X)
    loss = loss_fn(preds, y)
      
    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()
        
    if acc:
        return loss.item(), len(X), accuracy(preds, y)
    
    else:
        return loss.item(), len(X)
    
    
def fit(epochs, model, loss_fn, opt, train_dl, val_dl, device="cpu",
        stop_early=False, watch='val', patience=5, remix=False, spc=None):
    
    train_loss_epoch = []
    train_acc_epoch = []
    val_loss_epoch = []
    val_acc_epoch = []
    best_model = model
    best_loss = np.inf
    
    if stop_early:
        if watch == 'train': watch = train_loss_epoch
        elif watch == 'val': watch = val_loss_epoch
        else: raise ValueError('watch must be either "train" or "val"')
    
    model.to(device)
    for epoch in range(epochs):
        
        model.train()
        losses, nums, accs = zip(*[loss_batch(X.to(device),y.to(device), model, loss_fn, opt, acc=True, remix=remix, spc=spc) for X,y in tqdm(train_dl, desc=f'Epoch {epoch + 1} training')])
        train_loss = average(losses, nums)
        train_acc = average(accs, nums) 
            
        model.eval()
        with torch.no_grad():
            losses, nums, accs = zip(*[loss_batch(X.to(device), y.to(device), model, loss_fn, acc=True, remix=remix, spc=spc) for X, y in tqdm(val_dl, desc=f'Epoch {epoch + 1} testing')])

        val_loss = average(losses, nums)
        val_acc = average(accs, nums)   
        
        # update best model
        if val_loss < best_loss:
            best_model = deepcopy(model)
            best_loss = val_loss
        
        train_loss_epoch.append(train_loss)
        train_acc_epoch.append(train_acc)
        val_loss_epoch.append(val_loss)
        val_acc_epoch.append(val_acc)
        
        if stop_early and len(watch) >= patience:
            window = watch[-patience:]
            improvement = False
            for i in range(1, patience):
                if window[i] < window[0]:
                    improvement = True
                    break
            if not improvement:
                print(f'Early stopping after {epoch + 1} epochs')
                break
            
    return train_loss_epoch, train_acc_epoch, val_loss_epoch, val_acc_epoch, best_model



##### DATASETS #####

class TensorDatasetTransform(Dataset[Tuple[Tensor, Tensor]]):
    r"""Dataset wrapping tensors and optionally transforming samples.
    
    Each sample will be retrieved by indexing tensors along the first dimension.

    Args:
        tensors (Tuple): two tensors (features, labels) that have the same size of the first dimension.
        tranform (Callable | None): function that transforms each sample of first tensor.
        transform_probs (Tensor): probabilities of transformation for all samples.
    """

    def __init__(self, tensors: Tuple[Tensor, Tensor], 
                 transform: Callable[[Tensor, Tensor], Tensor] | None = None,
                 transform_probs: Tensor | None = None) -> None:
        
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors), "Size mismatch between tensors"
        self.tensors = tensors
        self.transform = transform
        self.trans_probs = transform_probs
        
        if self.transform:
            assert self.trans_probs is not None, "Transform_probs not provided, although transform was provided."
            
        # self.sampled_idxs = []
        # self.sampled_labels = []
        # self.sampled_transormed = []
        
    def __getitem__(self, index):
        feat, label = self.tensors[0][index], self.tensors[1][index]
        # self.sampled_idxs.append(index)
        # self.sampled_labels.append(label.item())
        if self.transform:
            if torch.bernoulli(self.trans_probs[index]):
                feat = self.transform(feat, label)
                # self.sampled_transormed.append(True)
            # else:
                # self.sampled_transormed.append(False)
        return feat, label

    def __len__(self):
        return self.tensors[0].size(0)



##### PLOTTING #####

def plot_cvae_hist(hist, total=True, rec=True, kl=True):
    fig, ax = plt.subplots(figsize=(12, 6))
    if total:
        ax.plot([l.item() for l in hist['train']['total']], label='total')
        ax.plot([l.item() for l in hist['test']['total']], label='test-total')
        
    if rec:
        ax.plot([l.item() for l in hist['train']['rec']], label='reconstruction')
        ax.plot([l.item() for l in hist['test']['rec']], label='test-reconstruction')
    if kl:
        ax.plot([l.item() for l in hist['train']['kl']], label='kl')
        ax.plot([l.item() for l in hist['test']['kl']], label='test-kl')
    ax.legend()
    # plt.show()
    return fig

# def plot_train_val(train_y, test_y, ylabel, extrema_fn=min):
#     epochs = len(train_y)
#     fig, ax = plt.subplots()
#     ax.plot(range(epochs), train_y, label='train')
#     ax.plot(range(epochs), test_y, label='test')
#     ax.legend()
#     ax.set_xticks(range(epochs))
#     ax.set_xlabel('epochs')
#     ax.set_ylabel(ylabel)
    
#     extr_train = extrema_fn(train_y)
#     epoch_train = train_y.index(extr_train)
#     extr_test = extrema_fn(test_y)
#     epoch_test = test_y.index(extr_test)

#     ax.plot(epoch_train, extr_train, 'ro')
#     ax.plot(epoch_test, extr_test, 'ro')
    
#     ax.axvline(x=epoch_train, color='gray', linestyle='--', alpha=0.5)
#     ax.axvline(x=epoch_test, color='gray', linestyle='--', alpha=0.5)

#     ax.annotate(f"{extr_train:.4f}", (epoch_train, extr_train), textcoords="offset points", xytext=(-10,-10), ha='center')
#     ax.annotate(f"{extr_test:.4f}", (epoch_test, extr_test), textcoords="offset points", xytext=(-10,-10), ha='center')

#     return fig

def plot_train_val(ax, train_y, test_y, ylabel, extrema_fn=min):
    epochs = len(train_y)
    # fig, ax = plt.subplots()
    ax.plot(range(epochs), train_y, label='train')
    ax.plot(range(epochs), test_y, label='val')
    ax.legend()
    ax.set_xticks(range(epochs))
    ax.set_xlabel('epochs')
    ax.set_ylabel(ylabel)
    
    extr_train = extrema_fn(train_y)
    epoch_train = train_y.index(extr_train)
    extr_val = extrema_fn(test_y)
    epoch_val = test_y.index(extr_val)

    ax.plot(epoch_train, extr_train, 'ro')
    ax.plot(epoch_val, extr_val, 'ro')
    
    ax.axvline(x=epoch_train, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=epoch_val, color='gray', linestyle='--', alpha=0.5)

    ax.annotate(f"{extr_train:.4f}", (epoch_train, extr_train), textcoords="offset points", xytext=(-10,-10), ha='center')
    ax.annotate(f"{extr_val:.4f}", (epoch_val, extr_val), textcoords="offset points", xytext=(-10,-10), ha='center')

    return ax


def plot_per_class_acc(ax, per_class, model_name):
    overall = np.array(per_class).mean()
    
    ax.bar(range(len(per_class)), per_class)
    # ax.set_xticks(range(len(per_class)))
    ax.set_ylabel('Accuracy')
    ax.set_xlabel('Class')
    ax.set_title(f"{model_name} overall: {round(overall, 2)}")
    # for i, acc in enumerate(per_class):
    #     ax.text(i, acc, f'{acc:.2f}', ha='center', va='bottom')    
    return ax

def plot_grouped_accs(ax, per_class, name, class_groups):
    overall = np.array(per_class).mean()
    heights = [np.array(per_class)[i].mean() for i in class_groups]

    ax.bar(range(len(class_groups)), heights)
    ax.set_ylabel('Accuracy')
    ax.set_xlabel('Class Groups')
    ax.set_title(f"{name}: {round(overall, 2)}")
    for i, height in enumerate(heights):
        ax.text(i, height, f'{round(height, 2)}', ha='center', va='bottom')
    return ax



##### GENERAL UTILS #####

def seed_everything(seed=42):
    """
    Ensure reproducibility.

    :param seed: Integer defining the seed number.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    
def samples_per_class(labels: np.ndarray):
    return torch.tensor([(l == labels).sum() for l in np.unique(labels)])
    
    
def evaluate(model, dl, num_classes, accuracy=True, report=False, cohen=False, f1=False, device='cpu'):
    assert any([accuracy, report, cohen, f1]), 'At least one evaluation metric has to be chosen.'
    
    model.eval()
    preds = []
    labels = []
    with torch.no_grad():
        for x,y in dl:
            preds.append(model(x.to(device)))
            labels.append(y)
    preds = torch.cat(preds).to(device)
    labels = torch.cat(labels).to(device)
    
    results = {}
    if accuracy: 
        accs = []
        for i in range(num_classes):
            idx = labels == i
            acc = (preds[idx].argmax(-1) == i).float().mean().item()
            accs.append(acc)
        
        overall_acc = (preds.argmax(-1) == labels).float().mean().item()
        results['accuracy'] = {'overall': overall_acc, 'per_class': accs}
    
    if report:
        results['report'] = classification_report(labels.cpu(), preds.argmax(dim=1).cpu(), output_dict=True)
        results['report_str'] = classification_report(labels.cpu(), preds.argmax(dim=1).cpu())
    
    if f1:    
        results['f1'] = f1_score(labels.cpu(), preds.argmax(dim=1).cpu(), average='weighted')
    
    if cohen:
        results['cohen'] = cohen_kappa_score(labels.cpu(), preds.argmax(dim=1).cpu())
    
    return results

def normalize(tensor):
    '''
    Normalize each feature of tensor with shape (samples, features).
    '''
    _min = tensor.min(dim=0).values
    _max = tensor.max(dim=0).values
    
    return (tensor - _min) / (_max - _min), _min, _max

def un_normalize(tensor, _min, _max, device='cpu'):
    return (tensor.to(device) * (_max.to(device) - _min.to(device))) + _min.to(device)


# standardization
# def normalize(tensor):
#     '''
#     Standardize each feature of tensor with shape (samples, features).
#     '''
#     mean = tensor.mean(dim=0)
#     std = tensor.std(dim=0)
    
#     return (tensor - mean) / std, mean, std             

# def un_normalize(tensor, mean, std, device='cpu'):
#     return tensor.to(device) * std.to(device) + mean.to(device)


# robust scaling
# def normalize(tensor):
#     '''
#     Robust scaling of each feature of tensor with shape (samples, features).
#     '''
#     median = tensor.median(dim=0).values
#     q1 = tensor.quantile(0.25, dim=0)
#     q3 = tensor.quantile(0.75, dim=0)
#     iqr = q3 - q1
    
#     return (tensor - median) / iqr, median, iqr

# def un_normalize(tensor, median, iqr, device='cpu'):
#     return tensor.to(device) * iqr.to(device) + median.to(device)


def normalize_min_max(tensor, _min, _max, device='cpu'):
    return (tensor.to(device) - _min.to(device)) / (_max.to(device) - _min.to(device))

# def sample_from_latent(cvae, n_per_class, num_classes, latent_dim, var=1.0):
#     cvae.eval()
#     gen_feats = torch.zeros((n_per_class * num_classes, 512))
#     gen_labels = torch.zeros((n_per_class * num_classes)).to(int)

#     for cls in range(num_classes):
#         latents = var * torch.randn((n_per_class, latent_dim)).to(device) # sample from standard normal distribution in latent space of cvae
#         x_hat = cvae.decoder(latents, F.one_hot(torch.full((n_per_class,),cls), num_classes=num_classes).to(device))
#         x_hat = un_normalize(x_hat, train_min, train_max)
#         gen_feats[cls*n_per_class : (cls+1)*n_per_class] = x_hat.detach()
#         gen_labels[cls*n_per_class : (cls+1)*n_per_class] = torch.full((n_per_class,), cls)

#     return gen_feats, gen_labels


def generate_ds(cvae_path, var, spc, device='cuda'):
    """
    Generates a dataset of synthetic samples using a trained CVAE model.

    Args:
        cvae_path (str): The path to the saved CVAE model.
        var (float): The variance of the latent space.
        spc (list): A list or array containing the number of samples to generate for each class.
        device (str, optional): The device to use for computation. Defaults to 'cuda'.

    Returns:
        tuple: A tuple containing the generated embeddings and labels.
            - gen_embeds (torch.Tensor): The generated embeddings.
            - gen_labels (torch.Tensor): The corresponding labels for the generated embeddings.
    """
    
    cvae = torch.load(os.path.join(cvae_path, 'cvae.pth'))
    num_classes = cvae.n_classes
    
    gen_embeds = []
    gen_labels = []

    for label in range(num_classes):
        to_gen = spc[label]
        
        latents = var * torch.randn((to_gen, cvae.latent_dim)).to(device)
        y_oh = F.one_hot(torch.tensor([label]), num_classes=num_classes).to(device)
        y_ohs = y_oh.repeat(to_gen, 1)
        with torch.no_grad():
            samples = cvae.decoder(latents, y_ohs)
        samples = un_normalize(samples, cvae.norm_min, cvae.norm_max)

        gen_embeds.append(samples)
        gen_labels.append(torch.tensor([label]*to_gen))
        
    gen_embeds = torch.cat(gen_embeds)
    gen_labels = torch.cat(gen_labels)
    
    return gen_embeds, gen_labels


##### TANSFORMTAIONS #####

def transform_gen_var(feat: Tensor, 
                      label: Tensor, 
                      var:float | int, 
                      num_classes: int, 
                      cvae: CVAE, 
                      device: str): 
    latent_dim = cvae.latent_dim
    norm = cvae.norm
    
    latents = var * torch.randn((latent_dim)).to(device)
    y_oh = F.one_hot(label, num_classes=num_classes).to(device)
    
    latents = latents.unsqueeze(0)
    y_oh = y_oh.unsqueeze(0)
    
    with torch.no_grad():
        x_hat = cvae.decoder(latents, y_oh)
    
    if norm:
        return un_normalize(x_hat, cvae.norm_min, cvae.norm_max, device=device).squeeze(0)
    else:
        return x_hat.squeeze(0)


# def transform_recon(feat: Tensor, label: Tensor):
#     y_oh = F.one_hot(label, num_classes=num_classes).to(device)
#     rec = cvae(normalize_min_max(feat, train_min.to(device), train_max.to(device)), y_oh)
#     return un_normalize(rec, train_min, train_max)
    
# def transform_interp(feat: Tensor, label: Tensor):
#     y_oh = F.one_hot(label, num_classes=num_classes).to(device)
#     rand_latents = 3 * torch.randn((latent_dim)).to(device)
#     z = cvae.encoder(normalize_min_max(feat, train_min.to(device), train_max.to(device)), y_oh)
#     interp = cvae.decoder(0.5 * z + 0.5 * rand_latents, y_oh)
#     return un_normalize(interp, train_min, train_max)

# def gen_var_min_norm(feat: Tensor, label: Tensor, var:float | int, min_norm: float):
#     norm = 0
#     while norm < min_norm:
#         latents = var * torch.randn((latent_dim)).to(device)
#         norm = torch.linalg.vector_norm(latents, ord=2, dim=0)
    
#     y_oh = F.one_hot(label, num_classes=num_classes).to(device)
#     x_hat = cvae.decoder(latents, y_oh)
#     return un_normalize(x_hat, train_min, train_max)

# def interp(feat: Tensor, label: Tensor, train_feats: Tensor, train_labels: Tensor):
#     feats_same_class = train_feats[train_labels == label]
#     # print(f'class: {label.item()}')
#     # print(f'{feats_same_class.shape=}')
#     rand_int = torch.randint(0, feats_same_class.size(0), (1,)).item()
#     # print(f'{rand_int=}')
#     second_feat = feats_same_class[rand_int]
#     # print(f'{second_feat.shape=}')
#     # if (feat == second_feat).all():
#     #     print(f'same, label: {label.item()}, rand_int: {rand_int}')
#     return 0.5 * feat + 0.5 * second_feat

# def gauss_noise_var(feat: Tensor, label: Tensor, var:float | int): 
#     noise = var * torch.randn((feat.shape)).to(device)
#     return feat + noise


##### DINO #####
def save_embeddings(model, ds, save_path, split, batch_size, device='cuda'):
    
    if not os.path.exists(os.path.join(save_path, split)):
        os.makedirs(os.path.join(save_path, split))
    else:
        print(f"Folder {os.path.join(save_path, split)} already exists. Exiting")
        return
    
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False)
    
    embeds_cls = []
    labels = []
    
    with torch.no_grad():
        for x, y in tqdm(dl, total=len(dl)):
            
            output = model.forward_features(x.to(device)) 
            output = model.forward_head(output, pre_logits=True)
            
            embeds_cls.append(output.detach().cpu())
            labels.append(y)
            # torch.save(z['x_norm_clstoken'], os.path.join(save_path, 'train', f'embeds_cls{i}.pt'))
            # torch.save(z['x_norm_patchtokens'], os.path.join(save_path, 'train', f'embeds_patches{i}.pt'))
            # torch.save(y, os.path.join(save_path, 'train', f'labels{i}.pt'))
            
    embeds_cls = torch.cat(embeds_cls)
    labels = torch.cat(labels)
    
    torch.save(embeds_cls, os.path.join(save_path, split, 'embeds_cls.pt'))
    torch.save(labels, os.path.join(save_path, split, 'labels.pt'))   