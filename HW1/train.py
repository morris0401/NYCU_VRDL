import os
import time

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Subset, default_collate
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from torchvision.models import resnext101_64x4d, ResNet
from torchvision.models.resnet import Bottleneck
from torchvision.ops import stochastic_depth
from torchvision.transforms import v2


def train_model(model, criterion, optimizer, scheduler,
                train_dataset, val_dataset, kf, num_epochs=25):
    """Train a model with k-fold cross-validation and return the best model.

    Args:
        model: The neural network model to train.
        criterion: Loss function.
        optimizer: Optimizer for training.
        scheduler: Learning rate scheduler.
        train_dataset: Dataset for training.
        val_dataset: Dataset for validation.
        kf: KFold object for cross-validation.
        num_epochs: Number of epochs to train (default: 25).

    Returns:
        The trained model with the best validation accuracy.
    """
    # Initialize training start time
    since = time.time()
    # Track the best validation accuracy
    best_accuracy = 0.0
    writer = SummaryWriter('./runs/resnext_experiment')

    external_val_loader = DataLoader(val_dataset,
                                     batch_size=64,
                                     shuffle=False,
                                     num_workers=4)

    for fold, (train_idx, val_idx) in enumerate(kf.split(train_dataset)):
        print(f'Fold {fold + 1}/{k_folds}')
        print('-' * 10)

        train_subset = Subset(train_dataset, train_idx)

        train_loader = DataLoader(
            train_subset,
            batch_size=32,
            shuffle=True,
            num_workers=4,
            collate_fn=collate_fn['train']
        )

        dataloaders = {
            'train': train_loader,
            'external_val': external_val_loader
        }

        for epoch in range(num_epochs):
            print(f'Epoch {epoch}/{num_epochs - 1}')
            print('-' * 10)

            for phase in ['train', 'external_val']:
                if phase == 'train':
                    model.train()
                else:
                    model.eval()

                running_loss = 0.0
                running_corrects = 0

                for inputs, labels in dataloaders[phase]:
                    inputs, labels = inputs.to(device), labels.to(device)
                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        if labels.dim() > 1:  # If labels are one-hot encoded
                            labels = torch.argmax(labels, dim=1)
                        loss = criterion(outputs, labels)

                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                if phase == 'train':
                    dataset_size = len(train_subset)
                else:
                    dataset_size = len(val_dataset)

                epoch_loss = running_loss / dataset_size
                epoch_acc = running_corrects.double() / dataset_size

                global_step = fold * num_epochs + epoch
                writer.add_scalar(f'{phase}/Loss/fold_{fold}',
                                  epoch_loss, global_step)
                writer.add_scalar(f'{phase}/Accuracy/fold_{fold}',
                                  epoch_acc, global_step)
                if phase == 'train':
                    writer.add_scalar('Learning_Rate',
                                      optimizer.param_groups[0]['lr'],
                                      global_step)

                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                if phase == 'external_val' and epoch_acc > best_accuracy:
                    best_accuracy = epoch_acc
                    torch.save(model.state_dict(), './best_model.pth')

            if phase == 'train':
                scheduler.step()
            print()

    time_elapsed = time.time() - since
    minutes = time_elapsed // 60
    second = time_elapsed // 60
    print(f'Training complete in {minutes:.0f}m {second % 60:.0f}s')
    print(f'Best external val Acc: {best_accuracy:.4f}')
    model.load_state_dict(torch.load('./best_model.pth'))
    return model


class StochasticDepthBottleneck(Bottleneck):
    """Bottleneck block with stochastic depth.

    Args:
        inplanes: Input channels.
        planes: Output channels.
        stride: Stride for convolution (default: 1).
        downsample: Downsample layer (default: None).
        groups: Number of groups for grouped convolution (default: 1).
        base_width: Base width for grouped convolution (default: 64).
        dilation: Dilation rate (default: 1).
        norm_layer: Normalization layer (default: None).
        drop_prob: Drop probability for stochastic depth (default: 0.2).
    """
    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 groups=1, base_width=64, dilation=1, norm_layer=None,
                 drop_prob=0.2):
        super(StochasticDepthBottleneck, self).__init__(
            inplanes, planes, stride=stride, downsample=downsample,
            groups=groups, base_width=base_width, dilation=dilation,
            norm_layer=norm_layer
        )
        self.drop_prob = drop_prob

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = stochastic_depth(out,
                               p=self.drop_prob,
                               mode="row",
                               training=self.training)

        out += identity
        out = self.relu(out)
        return out


def resnext101_64x4d_with_stochastic_depth(pretrained=False, drop_prob=0.2):
    """Create a ResNeXt-101 64x4d model with stochastic depth.

    Args:
        pretrained: If True, load pretrained weights (default: False).
        drop_prob: Drop probability for stochastic depth (default: 0.2).

    Returns:
        The ResNeXt-101 64x4d model with stochastic depth.
    """
    model = ResNet(StochasticDepthBottleneck,
                   [3, 4, 23, 3],
                   groups=64,
                   width_per_group=4)

    if pretrained:
        pretrained_model = resnext101_64x4d(weights='IMAGENET1K_V1')
        model.load_state_dict(pretrained_model.state_dict(), strict=False)

    for module in model.modules():
        if isinstance(module, StochasticDepthBottleneck):
            module.drop_prob = drop_prob

    return model


def set_drop_prob_linearly(model, drop_prob=0.5):
    """Set drop probabilities linearly for stochastic depth.

    Args:
        model: The model to set drop probabilities for.
        drop_prob: Maximum drop probability (default: 0.5).
    """
    total_blocks = sum([3, 4, 23, 3])
    current_block = 0
    for layer in [model.layer1, model.layer2, model.layer3, model.layer4]:
        for block in layer:
            block.drop_prob = (current_block / total_blocks) * drop_prob
            current_block += 1


def collate_fn_train(batch):
    """Custom collate function for training with data augmentation.

    Args:
        batch: Batch of data.

    Returns:
        Augmented batch.
    """
    return mixup(augmix(cutmix(*default_collate(batch))))


def collate_fn_val(batch):
    """Custom collate function for validation.

    Args:
        batch: Batch of data.

    Returns:
        Collated batch.
    """
    return default_collate(batch)


if __name__ == "__main__":
    cudnn.benchmark = True

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(324),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomGrayscale(0.2),
            transforms.ColorJitter(brightness=0.1, hue=0.1, saturation=0.1),
            transforms.RandomAffine(degrees=45,
                                    translate=(0.2, 0.2),
                                    scale=(0.8, 1.2),
                                    shear=(0.2, 0.2)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(360),
            transforms.CenterCrop(324),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    randaug = v2.RandomApply([v2.RandAugment()], p=0.3)
    augmix = v2.RandomApply([v2.AugMix()], p=0.5)
    cutmix = v2.RandomApply([v2.CutMix(num_classes=100)], p=0.5)
    mixup = v2.RandomApply([v2.MixUp(num_classes=100)], p=0.3)
    collate_fn = {'train': collate_fn_train, 'val': collate_fn_val}

    data_dir = './dataset/data'

    train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'),
                                         data_transforms['train'])
    val_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'),
                                       data_transforms['val'])

    k_folds = 10
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"Using {device} device")

    model_finetune = resnext101_64x4d_with_stochastic_depth(pretrained=True)
    set_drop_prob_linearly(model_finetune, drop_prob=0.5)

    num_ftrs = model_finetune.fc.in_features
    model_finetune.fc = nn.Sequential(
        nn.Linear(num_ftrs, len(train_dataset.classes)),
        nn.LogSoftmax())

    model_finetune = model_finetune.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer_ft = optim.AdamW(model_finetune.parameters(),
                               lr=1e-5,
                               weight_decay=1e-3)
    scheduler_ft = optim.lr_scheduler.CosineAnnealingLR(optimizer_ft,
                                                        T_max=15)

    model_finetune = train_model(model_finetune,
                                 criterion,
                                 optimizer_ft,
                                 scheduler_ft,
                                 train_dataset,
                                 val_dataset,
                                 kf,
                                 num_epochs=50
                                 )
    torch.save(model_finetune.state_dict(), './models/model.pth')
