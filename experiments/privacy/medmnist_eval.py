import os
import sys
import torch
import torch.nn.functional as F
from torch import Tensor
import matplotlib.pyplot as plt
import pickle
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from utils.utils import samples_per_class, fit, plot_train_val, evaluate, un_normalize, seed_everything, generate_ds
from models.models import N_Layer_Dense_Classifier


device = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_tensors(path, split):
    
    embeds = torch.load(os.path.join(path, split, 'embeds_cls.pt'))
    labels = torch.load(os.path.join(path, split, 'labels.pt')).squeeze()
    
    return embeds, labels


def load_data(embeds: Tensor, labels: Tensor, batch_size: int):
    
    ds = torch.utils.data.TensorDataset(embeds, labels)
    dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)
    
    return dl


def train_eval_classifier(train_dl, val_dl, test_dl, num_classes, save_path):
    
    os.makedirs(save_path, exist_ok=True)
    
    embs_model = N_Layer_Dense_Classifier(768, num_classes, num_layers=1)
    optim = torch.optim.Adam(embs_model.parameters(), lr=0.001)

    train_loss, train_acc, val_loss, val_acc, best_model = fit(epochs=200,
                                                            model=embs_model,
                                                            loss_fn=F.cross_entropy,
                                                            opt=optim,
                                                            train_dl=train_dl,
                                                            val_dl=val_dl,
                                                            device=device,
                                                            stop_early=True,
                                                            watch='val',
                                                            patience=10)

    fig, axs = plt.subplots(1, 2, figsize=(10,5))
    axs[0] = plot_train_val(axs[0], train_loss, val_loss, 'loss', extrema_fn=min)
    axs[1] = plot_train_val(axs[1], train_acc, val_acc, 'accuracy', extrema_fn=max)
    plt.savefig(os.path.join(save_path, 'train_history.png'))

    eval_results = evaluate(best_model, test_dl, num_classes, accuracy=True,
                            cohen=True, f1=True, device=device)

    with open(os.path.join(save_path, 'results.pkl'), 'wb') as pickle_file:
        pickle.dump(eval_results, pickle_file)
    

    

# def save_nn_distances(generated: Tensor, ground: Tensor, save_path: str, save_name='min_distances.pt', same=False):
    
#     os.makedirs(save_path, exist_ok=True)
    
#     distances = torch.cdist(generated.to('cuda'), ground.to('cuda'))
#     if same:
#         distances.fill_diagonal_(float('inf'))
    
#     min_distances = torch.min(distances, dim=1).values.to('cpu')
    
#     torch.save(min_distances, os.path.join(save_path, save_name))
    
#     del distances
#     del min_distances
#     torch.cuda.empty_cache()
    
def save_nn_distances(generated: Tensor, ground: Tensor, save_path: str, save_name='min_distances.pt', same=False, batch_size=1024):
    
    os.makedirs(save_path, exist_ok=True)
    
    num_generated = generated.size(0)
    
    min_distances = torch.empty(num_generated, device='cpu')
    
    generated = generated.to('cuda')
    ground = ground.to('cuda')
    
    for i in range(0, num_generated, batch_size):
        end_i = min(i + batch_size, num_generated)
        
        distances_batch = torch.cdist(generated[i:end_i], ground)
        
        if same:
            for j in range(end_i - i):
                distances_batch[j, i + j] = float('inf')
        
        min_distances[i:end_i] = torch.min(distances_batch, dim=1).values.to('cpu')
        
        del distances_batch
        torch.cuda.empty_cache()
        
    torch.save(min_distances, os.path.join(save_path, save_name))
    
    del min_distances
    torch.cuda.empty_cache()
    
    
def main():
    
    batch_size = 256
    
    seed_everything()
    
    data_dir_base = '/mnt/dtafler/test/medmnist3'
    cvae_path_base = 'experiments/privacy/trained_models'
    save_path_base = 'experiments/privacy/medmnist_eval_results4'

    ds_names = [ds for ds in os.listdir(data_dir_base) if os.path.isdir(os.path.join(data_dir_base, ds))]

    for ds_name in ds_names:
        print(f"\n\nProcessing {ds_name}...")
        
        data_path = os.path.join(data_dir_base, ds_name)
        cvae_path = os.path.join(cvae_path_base, ds_name)
        save_path = os.path.join(save_path_base, ds_name)
        
        # load original data
        train_embeds, train_labels = load_tensors(data_path, 'train')
        val_embeds, val_labels = load_tensors(data_path, 'val')
        test_embeds, test_labels = load_tensors(data_path, 'test')
        
        spc = samples_per_class(train_labels)
        num_classes = train_labels.unique().shape[0]
        
        for run in range(5):
        # train on original data
            train_dl = load_data(train_embeds, train_labels, batch_size)
            val_dl = load_data(val_embeds, val_labels, batch_size*2)
            test_dl = load_data(test_embeds, test_labels, batch_size*2)
            
            train_eval_classifier(train_dl, val_dl, test_dl, num_classes, os.path.join(save_path, f'run_{run}', 'classifier_original'))
            
            # train on generated data
            gen_embeds, gen_labels = generate_ds(cvae_path, 1, spc)
            gen_train_dl = load_data(gen_embeds, gen_labels, batch_size)
            
            train_eval_classifier(gen_train_dl, val_dl, test_dl, num_classes, os.path.join(save_path, f'run_{run}', 'classifier_generated'))
        
        # save distances
        save_nn_distances(gen_embeds, train_embeds, os.path.join(save_path, 'nn_distances'), 'generated_original.pt')
        save_nn_distances(train_embeds, train_embeds, os.path.join(save_path, 'nn_distances'), 'original_original.pt', same=True)
        
        torch.cuda.empty_cache()
        
if __name__ == '__main__':
    main()