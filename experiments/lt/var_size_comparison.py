import os
import torch
import pandas as pd
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from classifier import NetworkParams, Classifier
from utils.utils import seed_everything


def get_net_params(data_path):
    if 'cifar100' in data_path:
        num_classes = 100
    else: 
        num_classes = 10
    
    return NetworkParams(batch_size=256,
              lr=0.001,
              num_feats=768,
              num_classes=num_classes,
              epochs=10,
              early_stopping=False,
              save_only_best=False,
              data_path=data_path)    
    
def get_ours(data_path, cvae_path, cvae_var=1.0):
    return Classifier(get_net_params(data_path),
                   loss_fn='softmax',
                   sampling='class_balanced', 
                   use_cvae=True,
                    cvae_path=cvae_path,
                    cvae_var=cvae_var,
                    trans_probs='inverse')

def get_acc(model):
    model.train()
    model.eval_test()
    return model.get_results()[0]


if __name__ == '__main__':

    data_path = '/mnt/dtafler/test/timm/vit_base_patch14_dinov2.lvd142m/cifar10_0.01'
    save_path = './experiments/lt/comparison_var_size_results_small_var_cifar10_0.01'
    assert not os.path.exists(save_path), 'Save path already exists'

    for size in ['small', 'large']:
        
        cvae_path = f'./experiments/lt/trained_models/{size}/cifar10_0.01/cvae.pth'
        
        for cvae_var in [0.25, 0.5, 0.75]:
            
            for run in range(5):
                save_path_model = f'{save_path}/{size}/var_{cvae_var}/run_{run}'
                
                if not os.path.exists(save_path_model):
                    os.makedirs(save_path_model)
                    
                model = get_ours(data_path, cvae_path, cvae_var)
                
                acc = torch.tensor(get_acc(model))
                per_class = torch.tensor(model.test_results['accuracy']['per_class'])
                
                torch.save(acc, f'{save_path_model}/acc.pt')
                torch.save(per_class, f'{save_path_model}/per_class.pt')