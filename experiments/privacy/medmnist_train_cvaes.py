import medmnist
from medmnist import INFO, Evaluator
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import timm
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from utils.utils import seed_everything
from train_cvae import train_cvae


seed_everything()

device = 'cuda' if torch.cuda.is_available() else 'cpu'


data_dir_base = '/mnt/dtafler/test/medmnist3'
save_path_base = 'experiments/privacy/trained_models'

ds_names = [ds for ds in os.listdir(data_dir_base) if os.path.isdir(os.path.join(data_dir_base, ds))]

for ds_name in ds_names:
    print(f"\n\nProcessing {ds_name}...")
    
    data_path = os.path.join(data_dir_base, ds_name)
    save_path = os.path.join(save_path_base, ds_name)
    
    # get number of classes
    labels = torch.load(os.path.join(data_path, 'train', 'labels.pt'))
    num_classes = labels.unique().shape[0]
    
    model, hist, train_ds, test_ds = train_cvae(data_path=data_path,
                                            save_path=save_path,
                                            num_classes=num_classes,
                                            input_dim=768,
                                            epochs=500,
                                            hidden_dim=512,
                                            latent_dim=100,
                                            beta=0.01,
                                            small=False, 
                                            seed=42, 
                                            val=True)
    
