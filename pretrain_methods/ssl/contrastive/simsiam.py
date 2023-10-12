# https://github.com/facebookresearch/simsiam/

import torch.nn as nn
import torchvision.transforms as transforms


def simsiam_augmentation():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    # MoCo v2's aug: similar to SimCLR https://arxiv.org/abs/2002.05709
    augmentation = [
        transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([simsiam.loader.GaussianBlur([.1, 2.])], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ]
    
    return augmentation


def simsiam(backbone, projector, images, args):
    criterion = nn.CosineSimilarity(dim=1).cuda(args.gpu)
    
    if args.gpu is not None:
        images[0] = images[0].cuda(args.gpu, non_blocking=True)
        images[1] = images[1].cuda(args.gpu, non_blocking=True)

    # compute output and loss
    x1 = backbone(images[0])
    x2 = backbone(images[1])
    
    p1, p2, = projector(x1), projector(x2)
    z1, z2  = x1.detach(), x2.detach()

    loss = -(criterion(p1, z2).mean() + criterion(p2, z1).mean()) * 0.5

    return loss