import torch
import torch.nn.functional as F
from torch import nn


def balanced_softmax_loss(logits, labels, sample_per_class, reduction='mean'):
    """Compute the Balanced Softmax Loss between `logits` and the ground truth `labels`.
    Args:
      labels: A int tensor of size [batch].
      logits: A float tensor of size [batch, no_of_classes].
      sample_per_class: A int tensor of size [no of classes].
      reduction: string. One of "none", "mean", "sum"
    Returns:
      loss: A float tensor. Balanced Softmax Loss.
    """
    spc = sample_per_class.type_as(logits)
    spc = spc.unsqueeze(0).expand(logits.shape[0], -1)
    logits = logits + spc.log()
    loss = F.cross_entropy(input=logits, target=labels, reduction=reduction)
    return loss


def focal_loss(input, target, gamma=2, weight=None):
    ce_loss = F.cross_entropy(input, target, reduction='none', weight=weight)
    pt = torch.exp(-ce_loss)
    return ((1 - pt)**gamma * ce_loss).mean()
  
  
  
def get_eql_class_weights(samples_per_class, lambda_):
    freqs = samples_per_class / sum(samples_per_class)
    return (freqs > lambda_).float()

def replace_masked_values(tensor, mask, replace_with):
    assert tensor.dim() == mask.dim(), '{} vs {}'.format(tensor.shape, mask.shape)
    one_minus_mask = 1 - mask
    values_to_add = replace_with * one_minus_mask
    return tensor * mask + values_to_add

def softmax_eql(input, target, lambda_, ignore_prob, samples_per_class, device='cpu'):
    N, C = input.shape
    not_ignored = get_eql_class_weights(samples_per_class, lambda_).to(device).view(1, C).repeat(N, 1)
    over_prob = (torch.rand(input.shape).to(device) > ignore_prob).float()
    is_gt = target.new_zeros((N, C)).float()
    is_gt[torch.arange(N), target] = 1

    weights = ((not_ignored + over_prob + is_gt) > 0).float()
    input = replace_masked_values(input, weights, -1e7)
    loss = F.cross_entropy(input, target)
    return loss
  
  
class LDAMLoss(nn.Module):
    
    def __init__(self, cls_num_list, max_m=0.5, weight=None, s=30, device='cpu'):
        super(LDAMLoss, self).__init__()
        m_list = 1.0 / torch.sqrt(torch.sqrt(cls_num_list))
        m_list = m_list * (0.5 / torch.max(m_list))
        self.m_list = m_list.to(device)
        assert s > 0
        self.s = s
        self.weight = weight

    def forward(self, x, target):
        index = torch.zeros_like(x, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)
        
        index_float = index.type(torch.cuda.FloatTensor)
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0,1))
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m
    
        output = torch.where(index, x_m, x)
        return F.cross_entropy(self.s*output, target, weight=self.weight)