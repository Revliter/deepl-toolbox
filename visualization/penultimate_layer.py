import numpy as np
import torch
import torch.nn as nn

import copy
from typing import List
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt


def gram_schmidt(vectors):
    basis = []
    for v in vectors:
        w = v - np.sum( np.dot(v,b)*b  for b in basis )
        if (w > 1e-10).any():  
            basis.append(w/np.linalg.norm(w))
    return np.array(basis)


@torch.inference_mode()
def penultimate_layer_clus_visualize(model, dataloader, target_classes: List[int], impath:str, device:str):
    """This method is based on 'When Does Label Smoothing Help?' (arxiv 1906.02629v3)

    Args:
        model: the last layer of which is named 'linear'
        target_classes [list]: list of target classes
        dataloader : testloader or trainloader
        impath [str]: path to save img
        device [str]: cuda or cpu, the model and dataloader must be on the same device
    """
    
    model.eval()

    template_vecs = []
    for c in target_classes:
        template_vecs.append(model.linear.weight[c].detach().cpu().numpy())
    
    mlist = copy.deepcopy(model)
    mlist.linear = nn.Identity()
    
    # find orthonormal basis of plane using gram-schmidt
    basis = gram_schmidt(template_vecs)  # shape: (3, D)
    
    representations = []
    ys = []
    
    for x, y in dataloader:
        idx_to_use = []
        for idx in range(len(y)):
            if y[idx] in target_classes:
                idx_to_use.append(idx)
        
        if len(idx_to_use) == 0:
            continue

        x = x[idx_to_use].to(device)
        y = y[idx_to_use].to(device)

        representation = mlist(x).detach().cpu()

        for i in range(len(y)):
            representations.append(representation[i].numpy())
            ys.append(int(y[i].item()))
    
    X = np.stack(representations, axis=0)  # (N * 3, D)
    
    # visualize
    colors = ['blue', 'red', 'green']
    c = [colors[target_classes.index(y)] for y in ys]
    
    proj_X = X @ basis.T  # (N * 3, 3)
    
    proj_X_2d = PCA(n_components=2).fit_transform(proj_X)  # (N * 3, 2)
    plt.clf()
    plt.scatter(proj_X_2d[:, 0], proj_X_2d[:, 1], s=3, c=c)
    plt.savefig(impath)