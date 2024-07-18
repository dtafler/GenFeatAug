import os
import sys
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

sys.path.append('..')
from train_cvae import train_cvae
from models.models import N_Layer_Dense_Classifier
from utils.utils import fit




def run_exp(data_path, 
            save_path, 
            num_classes, 
            input_dim, 
            learning_rate, 
            weight_decay, 
            norm, 
            batch_size, 
            epochs, 
            hidden_dim, 
            latent_dim, 
            beta, 
            n):
    
    cvae_accs = []
    vae_accs = []
    
    cvae_recon_errors = []
    vae_recon_errors = []

    for run in range(n):
        # train models
        tmp_models_path = os.path.join(save_path, 'tmp_models')
        # remove tmp_models_path if it exists
        if os.path.exists(tmp_models_path):
            os.system(f'rm -r {tmp_models_path}')
            
        cvae, cvae_hist, train_ds, test_ds = train_cvae(data_path,
                                                    os.path.join(tmp_models_path, 'cvae'),
                                                    num_classes,
                                                    input_dim,
                                                    learning_rate,
                                                    weight_decay,
                                                    norm,
                                                    batch_size,
                                                    epochs,
                                                    hidden_dim,
                                                    latent_dim,
                                                    beta,
                                                    small=True)
        print(cvae)


        vae, vae_hist, _, _ = train_cvae(data_path,
                                    os.path.join(tmp_models_path, 'vae'),
                                    num_classes,
                                    input_dim,
                                    learning_rate,
                                    weight_decay,
                                    norm,
                                    batch_size,
                                    epochs,
                                    hidden_dim,
                                    latent_dim,
                                    beta,
                                    vae=True)
        print(vae)
        
        cvae_recon_errors.append(cvae_hist['test']['rec'][-1].item())
        vae_recon_errors.append(vae_hist['test']['rec'][-1].item())

        # save encodings of test set for vae and cvae
        test_dl = DataLoader(test_ds, batch_size=batch_size * 2, shuffle=False)

        cvae.eval()
        vae.eval()
        cvae_latents = []
        vae_latents = []
        labels = []
        for x, y in test_dl:
            x = x.cuda()
            y = y.cuda()
            y_oh = F.one_hot(y, num_classes)
            cvae_latents.append(cvae.encoder(x, y_oh))
            vae_latents.append(vae.encoder(x))
            labels.append(y)
            
        cvae_latents = torch.cat(cvae_latents, dim=0)
        vae_latents = torch.cat(vae_latents, dim=0)
        labels = torch.cat(labels, dim=0)

        # save for one run for later visualization
        if run == 0:
            torch.save(cvae_latents, os.path.join(save_path, 'cvae_latents.pt'))
            torch.save(vae_latents, os.path.join(save_path, 'vae_latents.pt'))
            torch.save(labels, os.path.join(save_path, 'labels.pt'))

        # train classifier
        cvae_accs.append(clsf_acc(cvae_latents, labels, latent_dim, num_classes))
        vae_accs.append(clsf_acc(vae_latents, labels, latent_dim, num_classes))
        
        
    cvae_accs = torch.tensor(cvae_accs)
    vae_accs = torch.tensor(vae_accs)
    cvae_recon_errors = torch.tensor(cvae_recon_errors)
    vae_recon_errors = torch.tensor(vae_recon_errors)
    
    torch.save(cvae_accs, os.path.join(save_path, 'cvae_accs.pt'))
    torch.save(vae_accs, os.path.join(save_path, 'vae_accs.pt'))
    torch.save(cvae_recon_errors, os.path.join(save_path, 'cvae_recon_errors.pt'))
    torch.save(vae_recon_errors, os.path.join(save_path, 'vae_recon_errors.pt'))


def clsf_acc(latents, labels, latent_dim, num_classes):

    classifier = N_Layer_Dense_Classifier(latent_dim, num_classes, num_layers=2)
    optim = torch.optim.Adam(classifier.parameters(), lr=0.01)

    X_train, X_test, y_train, y_test = train_test_split(latents.detach(), labels.detach(), test_size=0.2)
    train_ds = TensorDataset(X_train, y_train)
    test_ds = TensorDataset(X_test, y_test)

    _, _, _, test_accs, _ = fit(epochs=500,
                                model=classifier,
                                loss_fn=F.cross_entropy,
                                opt=optim,
                                train_dl=DataLoader(train_ds, batch_size=512, shuffle=True),
                                val_dl=DataLoader(test_ds, batch_size=512, shuffle=False),
                                device='cuda',
                                stop_early=True,
                                watch='train',
                                patience=10)
    print(test_accs)
    
    return test_accs[-1]


if __name__ == '__main__':
    
    data_path = '/mnt/dtafler/test/timm/vit_base_patch14_dinov2.lvd142m/cifar10_0.01'
    save_path = './cvae_latent'
    num_classes=10
    input_dim=768
    learning_rate=0.001
    weight_decay=0.01
    norm=True
    batch_size=512
    epochs=500
    hidden_dim=512
    latent_dim=2
    beta=0.1
    n = 5
    
    run_exp(data_path,
            save_path,
            num_classes,
            input_dim,
            learning_rate,
            weight_decay,
            norm,
            batch_size,
            epochs,
            hidden_dim,
            latent_dim,
            beta,
            n)

