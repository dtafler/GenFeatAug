from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import os
import torch
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
import torch.nn.functional as F
from tqdm import tqdm
from imblearn.over_sampling import SMOTE, ADASYN

from functools import partial
from typing import Iterable

from utils.utils import TensorDatasetTransform, samples_per_class, transform_gen_var, evaluate, fit, plot_grouped_accs, plot_train_val
from models.models import N_Layer_Dense_Classifier
import utils.losses as losses

from dataclasses import dataclass

@dataclass
class NetworkParams:
    batch_size: int
    lr: float
    num_feats: int
    num_classes: int
    epochs: int
    early_stopping: bool
    save_only_best: bool
    data_path: str


class Classifier:
    """
    Class for training a classifier with or without CVAE on long-tailed datasets without a validation set.
    
    Args:
        network_params (NetworkParams): An instance of the NetworkParams class containing network parameters.
        loss_fn (str): The loss function to be used for training the classifier.
        sampling (str): The sampling method to be used when training the model.
        use_cvae (bool, optional): Whether to use CVAE for data augmentation. Defaults to False.
        cvae_path (str, optional): The file path to the pre-trained CVAE model. Required if use_cvae is True.
        cvae_var (int, optional): The variance parameter for generating new samples using CVAE. Defaults to 1.
        trans_probs (str, optional): The probability distribution for applying transformations when using CVAE. 
                                    Defaults to 'inverse'. See self.get_trans_probs() for more options.
        device (str, optional): The device to be used for training the classifier. Defaults to 'cuda'.
    """
    
    def __init__(self, 
                 network_params: NetworkParams, 
                 loss_fn: str, 
                 sampling: str,
                 use_cvae: bool = False,
                 cvae_path: str = None,
                 cvae_var: int = 1,
                 trans_probs: str = 'inverse',
                 device: str = 'cuda'):
        
        assert sampling in ['class_balanced', 'instance_balanced', 'smote', 'adasyn', 'remix'], "Sampling method not supported"
        if sampling in ['smote', 'adasyn']:
            assert use_cvae == False, 'SMOTE, ADASYN, Remix not compatible with cvae'
        
        self.batch_size = network_params.batch_size
        self.lr = network_params.lr
        self.num_feats = network_params.num_feats
        self.num_classes = network_params.num_classes
        self.epochs = network_params.epochs
        self.early_stopping = network_params.early_stopping
        self.save_only_best = network_params.save_only_best
        self.data_path = network_params.data_path
        
        self.loss_fn = loss_fn
        
        self.sampling = sampling
        if self.sampling == 'remix' and not use_cvae:
            self.sampling = 'instance_balanced'
            self.remix = True
        elif self.sampling == 'remix' and use_cvae:
            self.sampling = 'class_balanced'
            self.remix = True
        else: 
            self.remix = False
        
        self.use_cvae = use_cvae
        self.cvae_path = cvae_path
        self.cvae_var = cvae_var
        self.trans_probs = trans_probs
        
        
        self.device = device
        
        if self.use_cvae:
            self.load_cvae()
            
        self.prep_data()
        self.load_data()
        self.load_loss_fn()
    
    
    def train(self):
        
        self.model = N_Layer_Dense_Classifier(self.num_feats, self.num_classes, num_layers=1).to(self.device)
        optim = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        
        self.loss_train, self.acc_train, self.loss_test, self.acc_test, self.best_model = fit(self.epochs, 
                                                                           self.model,
                                                                           self.loss_fn,
                                                                           optim,
                                                                           self.train_dl,
                                                                           self.test_dl,
                                                                           device=self.device,
                                                                           stop_early=self.early_stopping,
                                                                           watch='train',
                                                                           patience=3,
                                                                           remix=self.remix,
                                                                           spc=self.spc)
        
        if self.save_only_best:
            self.model = self.best_model


    def plot_train_val(self):
        
        fig, axs = plt.subplots(1, 2, figsize=(10,5))
        axs[0] = plot_train_val(axs[0], self.loss_train, self.loss_test, 'loss', extrema_fn=min)
        axs[1] = plot_train_val(axs[1], self.acc_train, self.acc_test, 'accuracy', extrema_fn=max)
        return fig
        
        
    def eval_test(self):
        
        self.test_results = evaluate(self.model, 
                                     self.test_dl, 
                                     self.num_classes, 
                                     accuracy=True, 
                                     cohen=True, 
                                     f1=True, 
                                     device=self.device)
        
        
    def print_test_results(self):
        
        print(f'accuracy: {self.test_results["accuracy"]["overall"]:.4f}')
        print(f'f1: {self.test_results["f1"]:.4f}')
        print(f'cohen\'s kappa: {self.test_results["f1"]:.4f}')
        
        
    def get_results(self):
        
        return self.test_results["accuracy"]["overall"], self.test_results["f1"], self.test_results["f1"]
    
    
    def plot_grouped_test_accs(self, class_groups: Iterable[Iterable[int]] = None):
        
        if class_groups is None:
            few_shot = self.spc <= 20
            med_shot = (100 >= self.spc) & (self.spc > 20)
            many_shot = self.spc > 100
            class_groups = [many_shot, med_shot, few_shot]
        
        fig, ax = plt.subplots()
        ax = plot_grouped_accs(ax, self.test_results['accuracy']['per_class'], 'average total', class_groups)
        return fig
        
        
    def load_loss_fn(self):
        
        assert self.loss_fn in ['softmax', 'balanced_softmax', 'class_balanced_softmax', 'focal', 'equalization', 'ldam'], "Loss function not supported"
        
        if self.loss_fn == 'softmax':
            self.loss_fn = F.cross_entropy
            
        elif self.loss_fn == 'balanced_softmax':
            self.loss_fn = partial(losses.balanced_softmax_loss, sample_per_class=self.spc, reduction='mean')
            
        elif self.loss_fn == 'class_balanced_softmax':
            self.loss_fn = losses.Loss(loss_type='cross_entropy',
                                       samples_per_class=self.spc,
                                       class_balanced=True,
                                       beta=0.99)
        elif self.loss_fn == 'focal':
            self.loss_fn = losses.focal_loss
            
        elif self.loss_fn == 'equalization':
            self.loss_fn = partial(losses.softmax_eql, 
                                   lambda_=1e-3, 
                                   ignore_prob=.95, 
                                   samples_per_class=self.spc,
                                   device=self.device)
        
        elif self.loss_fn == 'ldam':
            self.loss_fn = losses.LDAMLoss(self.spc, device=self.device)
        
        
    def load_cvae(self):
        
        self.cvae = torch.load(self.cvae_path)
        self.cvae.eval()
        
        
    def prep_data(self):
        
        self.train_feats = torch.load(os.path.join(self.data_path, 'train', 'embeds_cls.pt')).to(self.device)
        self.train_labels = torch.load(os.path.join(self.data_path, 'train', 'labels.pt')).to(self.device)

        self.test_feats = torch.load(os.path.join(self.data_path, 'test', 'embeds_cls.pt'))
        self.test_labels = torch.load(os.path.join(self.data_path, 'test', 'labels.pt'))
        
        # save the number of samples per class
        self.spc = samples_per_class(self.train_labels.cpu())
        
        # probability of transformation (discarding sample and generating new one) when using cvae
        if self.use_cvae:
            self.trans_probs = self.get_trans_probs()
        
            # function to generate new samples when using cvae
            self.transform = partial(transform_gen_var, 
                                var=self.cvae_var,
                                num_classes=self.num_classes,
                                cvae=self.cvae,
                                device=self.device)
        
        # weights for class-balanced sampling
        self.sample_weights = [1/self.spc[i] for i in self.train_labels]

    
    def load_data(self):
        
        if self.sampling == 'instance_balanced' and not self.use_cvae:
            train_ds = TensorDataset(self.train_feats, self.train_labels)
            test_ds = TensorDataset(self.test_feats, self.test_labels)
            
            self.train_dl = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
            self.test_dl = DataLoader(test_ds, batch_size=self.batch_size * 2, shuffle=False)
        
        
        elif self.sampling == 'instance_balanced' and self.use_cvae: 
            # dataset applies transformation (generating sample from CVAE latent space) according to probability trans_probs
            train_ds = TensorDatasetTransform((self.train_feats, self.train_labels), 
                                                   self.transform,
                                                   self.trans_probs)
            test_ds = TensorDataset(self.test_feats, self.test_labels)
            
            self.train_dl = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
            self.test_dl = DataLoader(test_ds, batch_size=self.batch_size * 2, shuffle=False)
            
            
        elif self.sampling == 'class_balanced' and not self.use_cvae:            
            train_ds = TensorDataset(self.train_feats, self.train_labels)
            test_ds = TensorDataset(self.test_feats, self.test_labels)
            
            sampler = WeightedRandomSampler(weights=self.sample_weights, 
                                            num_samples=len(train_ds), 
                                            replacement=True)
            
            self.train_dl = DataLoader(train_ds, 
                                       batch_size=self.batch_size, 
                                       sampler=sampler)
            
            self.test_dl = DataLoader(test_ds, batch_size=self.batch_size * 2, shuffle=False)
            
            
        elif self.sampling == 'class_balanced' and self.use_cvae:           
            train_ds = TensorDatasetTransform((self.train_feats, self.train_labels), 
                                                   self.transform,
                                                   self.trans_probs)
            test_ds = TensorDataset(self.test_feats, self.test_labels)
            
            sampler = WeightedRandomSampler(weights=self.sample_weights, 
                                            num_samples=len(train_ds), 
                                            replacement=True)
            
            self.train_dl = DataLoader(train_ds, 
                                       batch_size=self.batch_size, 
                                       sampler=sampler)
            self.test_dl = DataLoader(test_ds, batch_size=self.batch_size * 2, shuffle=False)
            
            
        elif self.sampling == 'smote':          
            neighbors = min(min(self.spc).item() - 1, 5)
            smote = SMOTE(random_state=42, k_neighbors=neighbors)
            self.train_feats, self.train_labels = smote.fit_resample(self.train_feats.cpu().numpy(), self.train_labels.cpu().numpy())
            self.train_feats, self.train_labels = torch.tensor(self.train_feats).to(self.device), torch.tensor(self.train_labels).to(self.device)
            
            print(f"{self.train_feats.shape=}, {self.train_labels.shape=}")
            
            train_ds = TensorDataset(self.train_feats, self.train_labels)
            test_ds = TensorDataset(self.test_feats, self.test_labels)
            
            self.train_dl = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
            self.test_dl = DataLoader(test_ds, batch_size=self.batch_size * 2, shuffle=False)
            
        elif self.sampling == 'adasyn':            
            neighbors = min(min(self.spc).item() - 1, 5)
            
            adasyn = ADASYN(random_state=42, n_neighbors=neighbors)
            self.train_feats, self.train_labels = adasyn.fit_resample(self.train_feats.cpu().numpy(), self.train_labels.cpu().numpy())
            self.train_feats, self.train_labels = torch.tensor(self.train_feats).to(self.device), torch.tensor(self.train_labels).to(self.device)
            
            print(f"{self.train_feats.shape=}, {self.train_labels.shape=}")
            
            train_ds = TensorDataset(self.train_feats, self.train_labels)
            test_ds = TensorDataset(self.test_feats, self.test_labels)
            
            self.train_dl = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
            self.test_dl = DataLoader(test_ds, batch_size=self.batch_size * 2, shuffle=False)
            
        
    def get_trans_probs(self):

        if self.trans_probs == 'inverse':
            weights = torch.Tensor(1/(self.spc))
            probs = ((weights - weights.min()) / weights.max() - weights.min())
            probs = torch.clamp_min(torch.clamp_max(probs, 0.7),0.01)
            
        elif self.trans_probs == 'inverse_steep':
            weights = torch.Tensor(1/(self.spc**2))
            probs = ((weights - weights.min()) / weights.max() - weights.min())
            probs = torch.clamp_min(torch.clamp_max(probs, 0.7),0.01)
            
        elif self.trans_probs == 'always':
            probs = torch.full_like(self.spc, 1).to(float)
        
        return torch.Tensor([probs[i] for i in self.train_labels])