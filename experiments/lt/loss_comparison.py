import os
import numpy as np
import pandas as pd
import torch
from functools import partial
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from classifier import NetworkParams, Classifier
from utils.utils import seed_everything


save_path = 'loss_comp_results'
if not os.path.exists(save_path):
    os.makedirs(save_path)

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
    
    
def get_loss_ours(data_path, cvae_path, loss, cvae_var=1.0):
    return Classifier(get_net_params(data_path),
                   loss_fn=loss,
                   sampling='class_balanced', 
                   use_cvae=True,
                   cvae_path=cvae_path,
                   cvae_var=cvae_var,
                   trans_probs='inverse')

    
def get_loss(data_path, loss):
    return Classifier(get_net_params(data_path),
                    loss_fn=loss,
                    sampling='instance_balanced')
    
 
def get_acc(model):
    model.train()
    model.eval_test()
    return model.get_results()[0]



if __name__ == '__main__':
    
    data_path_base = '/mnt/dtafler/test/timm/vit_base_patch14_dinov2.lvd142m'
    save_path = './experiments/lt/comparison_loss_cifar10'
    assert not os.path.exists(save_path), 'Save path already exists'
    
    for dataset in ['cifar10_0.01']:
        
        data_path = os.path.join(data_path_base, dataset) 
        cvae_path = f'./experiments/lt/trained_models/large/{dataset}/cvae.pth'
        
        losses = ['softmax', 'balanced_softmax', 'class_balanced_softmax', 'focal', 'equalization', 'ldam']
        
        for loss in losses:
            
            for model_name, model_getter in {'ours': partial(get_loss_ours, cvae_path=cvae_path), 'plain': get_loss}.items():
            
                for run in range(5):
                    
                    save_path_model = f'{save_path}/{dataset}/{model_name}/{loss}/run_{run}'
                    if not os.path.exists(save_path_model):
                        os.makedirs(save_path_model)
                        
                    model = model_getter(data_path, loss=loss)
                        
                    acc = torch.tensor(get_acc(model))
                    per_class = torch.tensor(model.test_results['accuracy']['per_class'])
                    
                    torch.save(acc, f'{save_path_model}/acc.pt')
                    torch.save(per_class, f'{save_path_model}/per_class.pt')
