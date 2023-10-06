import torch
import torch.nn as nn
import torchvision
from torchvision.models import resnet50, ResNet50_Weights

from .penultimate_layer import penultimate_layer_clus_visualize
from pretrained_models.resnet import prepare_model_and_dataloader


def test_penultimate_layer_clus_vis(device):
    model, trainloader, testloader = prepare_model_and_dataloader(load_exist=True)
    penultimate_layer_clus_visualize(model.module, trainloader, target_classes=[0, 1, 2], impath='./train_vis.png', device=device)
    penultimate_layer_clus_visualize(model.module, testloader, target_classes=[0, 1, 2], impath='./test_vis.png', device=device)


def test_grad_visualization(device):
    pass


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    test_penultimate_layer_clus_vis(device) # ok
    test_grad_visualization(device) # not tested


if __name__ == "__main__":
    main()