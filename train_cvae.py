import torch
from torch.utils.data import DataLoader, TensorDataset
from torch import Tensor
import os
import matplotlib.pyplot as plt
from utils.utils import normalize, cvae_fit, plot_cvae_hist, seed_everything, samples_per_class, TensorDatasetTransform
from models.models import CVAE, VAE, CVAEWithKLAnnealing, CVAESmall
import argparse
from imblearn.over_sampling import SMOTE
from functools import partial

def train_cvae(data_path, 
               save_path, 
               num_classes, 
               input_dim, 
               learning_rate=0.001, 
               weight_decay=0.01,
               norm=True, 
               batch_size=512, 
               epochs=150, 
               hidden_dim=256, 
               latent_dim=50, 
               beta=0.1, 
               cutmix=False, 
               smote=False,
               kl_annealing=False,
               vae=False, 
               class_weighted=False,
               noise=0,
               small=False,
               seed=None,
               val=False):
    if norm:
        print("\nNormalizing embeddings")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using {device}")
    
    if seed is not None:
        seed_everything(seed)

    assert not os.path.exists(save_path), f"cvae exists at {save_path}, aborting."
    os.makedirs(save_path)  


    # make cvae
    if vae:
        model = VAE(input_dim, hidden_dim, latent_dim, num_classes, norm=norm)
    else:
        if small:
            model = CVAESmall(input_dim, hidden_dim, latent_dim, num_classes, norm=norm)
        else:
            model = CVAE(input_dim, hidden_dim, latent_dim, num_classes, norm=norm)
            if kl_annealing:
                model = CVAEWithKLAnnealing(input_dim, 
                                            hidden_dim, 
                                            latent_dim, 
                                            num_classes,
                                            norm=norm,
                                            kl_start=0.01,
                                            kl_anneal_rate=0.001,
                                            kl_max=0.5)


    # load data
    train_path= os.path.join(data_path, 'train')
    if val:
        test_path = os.path.join(data_path, 'val')
    else:
        test_path = os.path.join(data_path, 'test')


    train_feats = torch.load(os.path.join(train_path, 'embeds_cls.pt'))
    test_feats = torch.load(os.path.join(test_path, 'embeds_cls.pt'))

    train_labels = torch.load(os.path.join(train_path, 'labels.pt'))
    test_labels = torch.load(os.path.join(test_path, 'labels.pt'))
    if train_labels.ndim == 2:
        train_labels = train_labels.squeeze()
    if test_labels.ndim == 2:
        test_labels = test_labels.squeeze()
        
    spc = samples_per_class(train_labels)

    if class_weighted:
        assert not smote, "Cannot use smote with class weighted loss"
        assert not cutmix, "Cannot use cutmix with class weighted loss"

    if smote:
        print('\n\nUsing smote...\n\n')
        neighbors = min(min(spc).item() - 1, 5)
        # print(min(spc).item())
        print(f'{neighbors=}')
        smote = SMOTE(random_state=42, k_neighbors=neighbors)
        print(f"BEFORE SMOTE: {train_feats.shape=}, {train_labels.shape=}")
        train_feats, train_labels = smote.fit_resample(train_feats.cpu().numpy(), train_labels.cpu().numpy())
        train_feats, train_labels = torch.tensor(train_feats).to(device), torch.tensor(train_labels).to(device)
        print(f"AFTER SMOTE: {train_feats.shape=}, {train_labels.shape=}")
        
    if noise > 0:
        
        def transform_noise(feat: Tensor, label: Tensor, max_noise: float, weights: Tensor):
            return feat + max_noise * weights[label] * torch.randn_like(feat)
            
        weights = torch.Tensor(1/(spc))
        weights = ((weights - weights.min()) / (weights.max() - weights.min()))
        
        transform_func = partial(transform_noise, max_noise=noise, weights=torch.clamp_min(weights, 0.1))
        
        probs = torch.clamp_min(torch.clamp_max(weights, 0.5),0.001)
        # transform_probs = (1/spc) / (1/spc).sum()
        plt.plot(probs)
        transform_probs = torch.Tensor([probs[i] for i in train_labels])

    if norm:
        train_feats_norm, train_min, train_max = normalize(train_feats)
        test_feats_norm, test_min, test_max = normalize(test_feats)
        
        model.norm_min = train_min
        model.norm_max = train_max

        torch.save(train_min, os.path.join(save_path, 'train_min.pt'))
        torch.save(train_max, os.path.join(save_path, 'train_max.pt'))
        
        if noise > 0:
            train_ds = TensorDatasetTransform((train_feats_norm, train_labels), transform_func, transform_probs)
        else:
            train_ds = TensorDataset(train_feats_norm, train_labels)
        test_ds = TensorDataset(test_feats_norm, test_labels)
        
    else:
        if noise > 0:
            train_ds = TensorDatasetTransform((train_feats, train_labels), transform_func, transform_probs)
        else:
            train_ds = TensorDataset(train_feats, train_labels)
        test_ds = TensorDataset(test_feats, test_labels)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=batch_size * 2, shuffle=False)


    # train cvae
    opt = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    hist = cvae_fit(model, opt, train_dl, test_dl=test_dl, device=device, 
                    beta=beta, epochs=epochs, cutmix=cutmix, vae=vae, 
                    class_weighted=class_weighted, spc=spc)


    # save
    if vae:
        torch.save(model, os.path.join(save_path, 'vae.pth'))
    else:
        torch.save(model, os.path.join(save_path, 'cvae.pth'))

    plot_cvae_hist(hist, kl=False, total=False).savefig(os.path.join(save_path, 'rec_hist.png'))
    plot_cvae_hist(hist, total=False, rec=False).savefig(os.path.join(save_path, 'kl_hist.png'))
    plot_cvae_hist(hist, rec=False, kl=False).savefig(os.path.join(save_path, 'total_hist.png'))
    
    return model, hist, train_ds, test_ds



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path', type=str, help='Path to the dataset')
    parser.add_argument('num_classes', type=int, help='Number of classes in the dataset')
    parser.add_argument('embedding_size', type=int, help='Size of the embeddings')
    parser.add_argument('save_path', type=str, help='Path to save the model')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--no_normalize', action='store_true', help='Whether to normalize the embeddings')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size')
    parser.add_argument('--epochs', type=int, default=150, help='Number of epochs')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden dimension of the CVAE')
    parser.add_argument('--latent_dim', type=int, default=50, help='Latent dimension of the CVAE')
    parser.add_argument('--beta', type=float, default=0.1, help='Beta parameter for the CVAE')
    parser.add_argument('--cutmix', action='store_true', help='Use cutmix')
    parser.add_argument('--smote', action='store_true', help='Use smote')
    args = parser.parse_args()
    
    train_cvae(args.data_path, 
               args.save_path, 
               args.num_classes, 
               args.embedding_size, 
               learning_rate=args.lr, 
               norm=not args.no_normalize, 
               batch_size=args.batch_size, 
               epochs=args.epochs, 
               hidden_dim=args.hidden_dim, 
               latent_dim=args.latent_dim, 
               beta=args.beta, 
               cutmix=args.cutmix, 
               smote=args.smote)
    