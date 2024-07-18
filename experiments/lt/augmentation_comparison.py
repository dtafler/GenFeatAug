import os
import pandas as pd
import sys
import torch
from functools import partial

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
              epochs=50,
              early_stopping=True,
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
    
def get_baseline(data_path):
    return Classifier(get_net_params(data_path),
                   loss_fn='softmax',
                   sampling='instance_balanced') 
    
def get_cbs(data_path):
    return Classifier(get_net_params(data_path),
                   loss_fn='softmax',
                   sampling='class_balanced')   
    
def get_smote(data_path):
    return Classifier(get_net_params(data_path),
                   loss_fn='softmax',
                   sampling='smote')
    
def get_adasyn(data_path):
    return Classifier(get_net_params(data_path),
                loss_fn='softmax',
                sampling='adasyn')
    
def get_remix(data_path):
    return Classifier(get_net_params(data_path),
                   loss_fn='softmax',
                   sampling='remix')   

def get_ours_remix(data_path, cvae_path, cvae_var=1.0):
    return Classifier(get_net_params(data_path),
                   loss_fn='softmax',
                   sampling='remix', 
                   use_cvae=True,
                    cvae_path=cvae_path,
                    cvae_var=cvae_var,
                    trans_probs='inverse')    

def get_acc(model):
    model.train()
    model.eval_test()
    return model.get_results()[0]

    
if __name__ == "__main__":
    
    data_path_base = '/mnt/dtafler/test/timm/vit_base_patch14_dinov2.lvd142m'
    save_path = './experiments/lt/comparison_augmentation_results_early_stopping'
    
    
    for dataset in ['cifar100_0.01', 'cifar10_0.01']:
        
        data_path = os.path.join(data_path_base, dataset) 
        cvae_path = f'./experiments/lt/trained_models/large/{dataset}/cvae.pth'
        
        models = {
            'ours': partial(get_ours, cvae_path=cvae_path),
            'ours_remix': partial(get_ours_remix, cvae_path=cvae_path),
            'baseline': get_baseline,
            'cbs': get_cbs,
            'smote': get_smote,
            'adasyn': get_adasyn,
            'remix': get_remix
        }   
            
        for model_name, model_getter in models.items():
            
            for run in range(5):
                
                save_path_model = f'{save_path}/{dataset}/{model_name}/run_{run}'
                if not os.path.exists(save_path_model):
                    os.makedirs(save_path_model)
                    
                model = model_getter(data_path)
                    
                acc = torch.tensor(get_acc(model))
                per_class = torch.tensor(model.test_results['accuracy']['per_class'])
                
                torch.save(acc, f'{save_path_model}/acc.pt')
                torch.save(per_class, f'{save_path_model}/per_class.pt')