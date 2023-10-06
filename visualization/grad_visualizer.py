import torch
import torch.nn as nn
import torch.optim as optim

import copy
from typing import List
import matplotlib.pyplot as plt


def get_g(model):
    tensors = [p.grad.data.flatten() for n, p in model.named_parameters() if p.requires_grad]
    return torch.cat(tensors)


def get_w(model):
    tensors = [p.data.flatten() for n, p in model.named_parameters() if p.requires_grad]
    return torch.cat(tensors)


def plot_grad(x, it, t, dataset):
    from matplotlib import pyplot as plt
    # hist
    plt.clf()
    plt.hist(x.cpu().numpy(), bins=100)
    plt.savefig(f"./gimages/hist/{dataset}_{it}_{t}.png")
    # plot
    plt.clf()
    plt.plot(x.cpu().numpy())
    plt.savefig(f"./gimages/plot/{dataset}_{it}_{t}.png")


def get_loss_g(loss, optimizer):
    loss.backward(retain_graph=True)
    g = get_g(model)
    optimizer.zero_grad()
    return g


def loss_fn(logits, y):
    return F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-1, reduction='none')


@torch.inference_mode()
def compute_bp_grad(model, dataloader):
    model.eval()
    mlist = copy.deepcopy(model)
    mlist.linear = nn.Identity()
    
    train_data = []
    ys = []
    optimizer = optim.SGD(mlist.parameters(), lr=1e-2, momentum=0.9)
    
    for x, y in dataloader:
        
        optimizer.zero_grad()
        
        x = x.to(device)
        y = y.to(device)
        
        result = mlist(x)
        loss = loss_fn(result, y)
        grads = get_loss_g(torch.mean(loss), optimizer)  
        plot(grads, iter_num, 'gt')
        
        ys += [y]