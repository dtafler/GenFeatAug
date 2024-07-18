import medmnist
from medmnist import INFO, Evaluator
import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms as T
from tqdm import tqdm
import timm
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from utils.utils import seed_everything, save_embeddings

seed_everything()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# load dino model
model = 'vit_base_patch14_dinov2.lvd142m'
model = timm.create_model(model, pretrained=True, num_classes=0).to(device)
model.requires_grad_(False)
model = model.eval()

# get transforms for model
data_config = timm.data.resolve_model_data_config(model)
transforms = timm.data.create_transform(**data_config, is_training=False)
convert_to_rgb = T.Lambda(lambda img: img.convert('RGB') if img.mode == 'L' else img)
transform_list = [convert_to_rgb] + list(transforms.transforms)
transforms = T.Compose(transform_list)


# Set the directory to save the datasets
data_dir = '/mnt/dtafler/test/medmnist3'
os.makedirs(data_dir)

# Loop through all relevant MedMNIST datasets 
for dataset_name, info in INFO.items():
    if info['task'] == 'multi-class' and '3d' not in dataset_name:
        print(f"Downloading {dataset_name}...")
        info = INFO[dataset_name]
        
        # Download the dataset
        DataClass = getattr(medmnist, info['python_class'])
        train_ds = DataClass(split='train', download=True, root=data_dir, size=224)
        val_ds = DataClass(split='val', download=True, root=data_dir, size=224)
        test_ds = DataClass(split='test', download=True, root=data_dir, size=224)
        
        train_ds.transform = transforms
        val_ds.transform = transforms
        test_ds.transform = transforms
        
        save_path = os.path.join(data_dir, dataset_name)
        save_embeddings(model, test_ds, save_path, split='test', batch_size=32)
        save_embeddings(model, val_ds, save_path, split='val', batch_size=32)
        save_embeddings(model, train_ds, save_path, split='train', batch_size=32)
        
        
