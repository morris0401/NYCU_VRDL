import random
import torch
import numpy as np
from tqdm import tqdm


def set_seed(seed=77):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def collate_fn(batch):
    return tuple(zip(*batch))


def get_optimizer(model):
    backbone_body_params = []
    fpn_params = []
    rpn_params = []
    roi_params = []

    for name, param in model.named_parameters():
        if 'backbone.body' in name:
            backbone_body_params.append(param)
        elif 'backbone.fpn' in name:
            fpn_params.append(param)
        elif 'rpn' in name:
            rpn_params.append(param)
        elif 'roi_heads' in name:
            roi_params.append(param)

    optimizer = torch.optim.AdamW([
        {'params': backbone_body_params, 'lr': 0.00003},
        {'params': fpn_params, 'lr': 0.00015},
        {'params': rpn_params, 'lr': 0.0003},
        {'params': roi_params, 'lr': 0.00045},
    ])

    return optimizer


def train(model, optimizer, data_loader, device, scaler):
    model.train()
    running_loss = 0.0
    pbar = tqdm(data_loader, desc="Training", leave=False)
    for images, targets in pbar:
        images = list(img.to(device) for img in images)
        targets = [{k: v.to(device)
                   for k, v in t.items()}
                   for t in targets]

        optimizer.zero_grad()

        with torch.amp.autocast('cuda'):
            loss_dict = model(images, targets)
            losses = sum(loss
                         for loss
                         in loss_dict.values())

        scaler.scale(losses).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += losses.item()
        pbar.set_postfix(loss=losses.item())

    return running_loss / len(data_loader)
