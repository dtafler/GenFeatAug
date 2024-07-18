import os
import sys
import torch
import torch.nn.functional as F
from torch import Tensor
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import numpy as np
from scipy.linalg import sqrtm
import csv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from utils.utils import samples_per_class, fit, plot_train_val, evaluate, un_normalize, seed_everything
from models.models import N_Layer_Dense_Classifier


def generate_ds(cvae_path, var, spc):
    
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


def normalize_embeddings(embeddings):

    embeddings = embeddings.astype(np.float64)
    mean = np.mean(embeddings, axis=0)
    std = np.std(embeddings, axis=0)

    return (embeddings - mean) / std


# def calculate_statistics(vectors):
    
#     mu = np.mean(vectors, axis=0)
#     sigma = np.cov(vectors, rowvar=False)
    
#     return mu, sigma
def calculate_statistics(vectors):

    vectors = normalize_embeddings(vectors)
    mu = np.mean(vectors, axis=0)
    sigma = np.cov(vectors, rowvar=False)

    return mu, sigma


def calculate_fid(mu1, sigma1, mu2, sigma2):
    mu1, sigma1, mu2, sigma2 = map(np.float64, [mu1, sigma1, mu2, sigma2])
    
    diff = mu1 - mu2
    
    # Add a small value to the diagonal of the covariance matrices for numerical stability
    epsilon = 1e-6
    sigma1 += epsilon * np.eye(sigma1.shape[0])
    sigma2 += epsilon * np.eye(sigma2.shape[0])
    
    covmean, err = sqrtm(sigma1 @ sigma2, disp=False)
    print(f'estimated sqrtm error: {err}')
    
    # Numerical stability: if covmean contains complex numbers, only the real part is used
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
    
    # Ensure FID is non-negative due to numerical issues
    if fid < 0:
        fid = 0.0
    
    return fid


def main():
    
    cvae_base_path = './experiments/embeddings/fid_trained_models'
    data_base_path = '/mnt/dtafler/test/timm/vit_base_patch14_dinov2.lvd142m'
    
    results = []    

    for model_size in ['small', 'large']:
        print(f'\n\nProcessing {model_size} models')
        for dataset_name in ['cifar10_0.01', 'cifar100_0.01', 'cifar10_bal', 'cifar100_bal']:
            print(f'\nProcessing {dataset_name}')
            for var in [0.25, 0.5, 0.75, 1.0, 2, 3]:
                print(f'\nProcessing var {var}')
                
                data_path = os.path.join(data_base_path, dataset_name)
                test_embeds = torch.load(os.path.join(data_path, 'test', 'embeds_cls.pt'))
                test_labels = torch.load(os.path.join(data_path, 'test', 'labels.pt'))
                
                cvae_path = os.path.join(cvae_base_path, model_size, dataset_name)
                spc = samples_per_class(test_labels)
                gen_embeds, gen_labels = generate_ds(cvae_path, var, spc)
                print(f'After Generation:\n{gen_embeds.shape=}')
                print(f'{test_embeds.shape=}\n\n')
                
                # mu1, sigma1 = calculate_statistics(test_embeds.cpu().numpy())
                # mu2, sigma2 = calculate_statistics(gen_embeds.cpu().numpy())
                
                # fid = calculate_fid(mu1, sigma1, mu2, sigma2)     
                
                class_fid_scores = []

                for cls in torch.unique(test_labels):
                    print(f'class {cls}')
                    real_cls_embeds = test_embeds[test_labels == cls]
                    gen_cls_embeds = gen_embeds[gen_labels == cls]
                    print(f'\t{real_cls_embeds.shape=}')
                    print(f'\t{gen_cls_embeds.shape=}')
                    
                    if len(real_cls_embeds) == 0 or len(gen_cls_embeds) == 0:
                        continue

                    mu1, sigma1 = calculate_statistics(real_cls_embeds.cpu().numpy())
                    mu2, sigma2 = calculate_statistics(gen_cls_embeds.cpu().numpy())
                    
                    fid = calculate_fid(mu1, sigma1, mu2, sigma2)
                    print(f'\t{fid=}')
                    class_fid_scores.append(fid)
                
                mean_fid = np.mean(class_fid_scores) if class_fid_scores else float('nan')
                
                results.append({
                    'model_size': model_size,
                    'dataset_name': dataset_name,
                    'var': var,
                    'fid': mean_fid
                })      
                
    csv_file = './experiments/embeddings/fid_results_64.csv'
    with open(csv_file, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['model_size', 'dataset_name', 'var', 'fid'])
        writer.writeheader()
        writer.writerows(results)   


if __name__ == '__main__':
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    main()