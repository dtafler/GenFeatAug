import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from train_cvae import train_cvae

def train_cvaes(data_paths: dict, save_path: str):
    
    for name, data_path in data_paths.items():
        
        num_classes = 100 if 'cifar100' in data_path else 10
        _, _, _, _ = train_cvae(data_path=data_path,
                                save_path=os.path.join(save_path, 'large', name),
                                num_classes=num_classes,
                                input_dim=768,
                                epochs=500,
                                hidden_dim=512,
                                latent_dim=100,
                                beta=0.01,
                                small=False, 
                                seed=42)
        
        _, _, _, _ = train_cvae(data_path=data_path,
                                save_path=os.path.join(save_path, 'small', name),
                                num_classes=num_classes,
                                input_dim=768,
                                epochs=500,
                                hidden_dim=512,
                                latent_dim=100,
                                beta=0.01,
                                small=True, 
                                seed=42)

    

if __name__ == "__main__":
    
    data_paths = {
        'cifar10_0.1': '/mnt/dtafler/test/timm/vit_base_patch14_dinov2.lvd142m/cifar10_0.1',
        'cifar10_0.01': '/mnt/dtafler/test/timm/vit_base_patch14_dinov2.lvd142m/cifar10_0.01',
        'cifar100_0.1': '/mnt/dtafler/test/timm/vit_base_patch14_dinov2.lvd142m/cifar100_0.1',
        'cifar100_0.01': '/mnt/dtafler/test/timm/vit_base_patch14_dinov2.lvd142m/cifar100_0.01'
    }
    
    save_path = './experiments/lt/trained_models'
    
    train_cvaes(data_paths, save_path)