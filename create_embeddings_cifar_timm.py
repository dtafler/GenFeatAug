import torch
from torchvision.datasets import CIFAR10, CIFAR100
from torch.utils.data import DataLoader
import os
import argparse
from tqdm import tqdm
from typing import Iterable
from utils.utils import seed_everything
import timm


seed_everything()


parser = argparse.ArgumentParser()
parser.add_argument('save_path', type=str, help='Path to save the embeddings')
parser.add_argument('model', type=str, default='vit_base_patch14_dinov2.lvd142m', help='DINO model to use')
parser.add_argument('cifar10_or_100', type=int, default=10, help='Number of classes to use')
parser.add_argument('imbalance_ratio', type=float, default=0.01, help='Imbalance ratio')
parser.add_argument('batch_size', type=int, default=32, help='Batch size for dataloader')
parser.add_argument('--no_imb', action='store_true', help='do not create imbalance')

args = parser.parse_args()
assert args.cifar10_or_100 in [10, 100], "cifar10_or_100 must be 10 or 100"

model = args.model
num_classes = args.cifar10_or_100
imbalance_ratio = args.imbalance_ratio
batch_size = args.batch_size

if num_classes == 10:
    class_size = 5000
else:
    class_size = 500

save_path = args.save_path

assert not os.path.exists(save_path), f"embeddings exist at {save_path}, aborting."

print(f"Saving embeddings to {save_path}")
os.makedirs(os.path.join(save_path, 'train'))
os.makedirs(os.path.join(save_path, 'test'))

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using {device}")

# load model
model = timm.create_model(model, pretrained=True, num_classes=0).to(device)
model.requires_grad_(False)
model = model.eval()

# get transforms for model
data_config = timm.data.resolve_model_data_config(model)
transforms = timm.data.create_transform(**data_config, is_training=False)


# load data
if num_classes == 10:
    cifar_train = CIFAR10(root='../data', train=True, download=True) #TODO change data path
    cifar_test = CIFAR10(root='../data', train=False, download=True)
elif num_classes == 100:
    cifar_train = CIFAR100(root='../data', train=True, download=True)
    cifar_test = CIFAR100(root='../data', train=False, download=True)

def get_imbalanced_class_sizes(n_classes: int,
                               class_size: int, 
                               imbalance_ratio: float):
    samples_per_class = []
    for cls_idx in range(n_classes):
        num = class_size * (imbalance_ratio ** (cls_idx / (n_classes - 1)))
        samples_per_class.append(int(num))
    return samples_per_class

def imbalance(data: torch.tensor, labels: torch.tensor, spc: Iterable):
    new_data = []
    new_labels = []
    for cls_idx, num in enumerate(spc):
        idxs = (labels == cls_idx).nonzero(as_tuple=True)[0]
        idxs = idxs[torch.randperm(idxs.size(0))]
        new_data.append(data[idxs[:num]])
        new_labels.append(labels[idxs[:num]])
    return torch.cat(new_data), torch.cat(new_labels)


# imbalance data
dat = torch.tensor(cifar_train.data)
labels = torch.tensor(cifar_train.targets)
if not args.no_imb:
    spc = get_imbalanced_class_sizes(num_classes, class_size, imbalance_ratio)
    cifar_imb, cifar_imb_labels = imbalance(dat, labels, spc)
else: 
    cifar_imb = dat
    cifar_imb_labels = labels
print(f"\n\n{cifar_imb.size=}, {cifar_imb_labels.size=}")

cifar_train.data = cifar_imb.numpy()
cifar_train.targets = cifar_imb_labels.tolist()
cifar_train.transform = transforms
cifar_test.transform = transforms

train_dl = DataLoader(cifar_train, batch_size=batch_size, shuffle=False)
test_dl = DataLoader(cifar_test, batch_size=batch_size, shuffle=False)


print("Generating training embeddings")
train_embeds_cls = []
train_labels = []
with torch.no_grad():
    for i, (x, y) in tqdm((enumerate(train_dl)), total=len(train_dl)):
        output = model.forward_features(x.to(device)) 
        output = model.forward_head(output, pre_logits=True)
        train_embeds_cls.append(output.detach().cpu())
        # train_embeds_patches.append(z['x_norm_patchtokens'].cpu())
        train_labels.append(y)
        # torch.save(z['x_norm_clstoken'], os.path.join(save_path, 'train', f'embeds_cls{i}.pt'))
        # torch.save(z['x_norm_patchtokens'], os.path.join(save_path, 'train', f'embeds_patches{i}.pt'))
        # torch.save(y, os.path.join(save_path, 'train', f'labels{i}.pt'))
train_embeds_cls = torch.cat(train_embeds_cls)
train_labels = torch.cat(train_labels)
torch.save(train_embeds_cls, os.path.join(save_path, 'train', 'embeds_cls.pt'))
torch.save(train_labels, os.path.join(save_path, 'train', 'labels.pt'))


print("Generating testing embeddings")
test_embeds_cls = []
test_labels = []
with torch.no_grad():
    for i, (x, y) in tqdm(enumerate(test_dl), total=len(test_dl)):
        output = model.forward_features(x.to(device)) 
        output = model.forward_head(output, pre_logits=True)
        test_embeds_cls.append(output.detach().cpu())
        # test_embeds_patches.append(z['x_norm_patchtokens'].cpu())
        test_labels.append(y)
        # torch.save(z['x_norm_clstoken'], os.path.join(save_path, 'test', f'embeds_cls{i}.pt'))
        # torch.save(z['x_norm_patchtokens'], os.path.join(save_path, 'test', f'embeds_patches{i}.pt'))
        # torch.save(y, os.path.join(save_path, 'test', f'labels{i}.pt'))
test_embeds_cls = torch.cat(test_embeds_cls)
test_labels = torch.cat(test_labels)
torch.save(test_embeds_cls, os.path.join(save_path, 'test', 'embeds_cls.pt'))
torch.save(test_labels, os.path.join(save_path, 'test', 'labels.pt'))