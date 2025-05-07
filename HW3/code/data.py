import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2
import skimage.io as sio
import scipy.ndimage
import math

PADDING_VALUE = 400


def pad_image_to_size(image,
                      masks=None,
                      boxes=None,
                      target_size=(PADDING_VALUE, PADDING_VALUE)
                      ):
    target_height, target_width = target_size
    orig_width, orig_height = image.size
    orig_size = (orig_height, orig_width)

    pad_left = (target_width - orig_width) // 2 if \
        orig_width < target_width else 0
    pad_right = target_width - orig_width - pad_left if \
        orig_width < target_width else 0
    pad_top = (target_height - orig_height) // 2 if \
        orig_height < target_height else 0
    pad_bottom = target_height - orig_height - pad_top if \
        orig_height < target_height else 0

    if pad_left == 0 and pad_right == 0 and pad_top == 0 and pad_bottom == 0:
        return image, masks, boxes, orig_size

    padded_image = Image.new('RGB',
                             (orig_width + pad_left + pad_right,
                              orig_height + pad_top + pad_bottom),
                             (0, 0, 0))
    padded_image.paste(image, (pad_left, pad_top))

    padded_masks = masks
    padded_boxes = boxes

    if masks is not None:
        padded_masks = []
        for mask in masks:
            mask_np = mask.numpy()
            padded_mask = np.pad(
                mask_np,
                ((pad_top, pad_bottom), (pad_left, pad_right)),
                mode='constant',
                constant_values=0
            )
            padded_masks.append(torch.as_tensor(padded_mask,
                                                dtype=torch.uint8))

    if boxes is not None:
        padded_boxes = boxes.clone()
        padded_boxes[:, 0] += pad_left  # xmin
        padded_boxes[:, 2] += pad_left  # xmax
        padded_boxes[:, 1] += pad_top   # ymin
        padded_boxes[:, 3] += pad_top   # ymax

    return padded_image, padded_masks, padded_boxes, orig_size


def split_image_into_patches(image):
    orig_width, orig_height = image.size
    orig_size = (orig_height, orig_width)

    num_w = math.ceil(orig_width / PADDING_VALUE)
    num_h = math.ceil(orig_height / PADDING_VALUE)

    sub_images = []
    offsets = []

    for j in range(num_h):
        for i in range(num_w):
            x_start = i * PADDING_VALUE
            y_start = j * PADDING_VALUE
            x_end = min(x_start + PADDING_VALUE, orig_width)
            y_end = min(y_start + PADDING_VALUE, orig_height)

            sub_img = image.crop((x_start, y_start, x_end, y_end))
            sub_images.append(sub_img)
            offsets.append((x_start, y_start))

    return sub_images, offsets, orig_size


class SegmentationDataset(Dataset):
    def __init__(self, folders, transforms=None):
        self.folders = folders
        self.transforms = transforms
        self.samples = []
        for idx, folder in enumerate(folders):
            image_path = os.path.join(folder, 'image.tif')
            img = Image.open(image_path).convert("RGB")
            sub_images, offsets, _ = split_image_into_patches(img)
            for sub_idx, (sub_img, offset) in \
                    enumerate(zip(sub_images, offsets)):
                self.samples.append({
                    'folder': folder,
                    'image': sub_img,
                    'offset': offset,
                    'image_idx': idx,
                    'sub_idx': sub_idx
                })

    def __getitem__(self, idx):
        sample = self.samples[idx]
        folder = sample['folder']
        img = sample['image']
        x_offset, y_offset = sample['offset']
        image_idx = sample['image_idx']

        masks = []
        labels = []
        for i in range(1, 5):
            mask_path = os.path.join(folder, f'class{i}.tif')
            if os.path.exists(mask_path):
                mask = sio.imread(mask_path)
                mask = np.array(mask) > 0
                labeled_mask, num_features = scipy.ndimage.label(mask)
                for j in range(1, num_features + 1):
                    component_mask = (labeled_mask == j)

                    y_end = min(y_offset + img.height, mask.shape[0])
                    x_end = min(x_offset + img.width, mask.shape[1])
                    component_mask = component_mask[y_offset:y_end,
                                                    x_offset:x_end]
                    if component_mask.shape != (y_end - y_offset,
                                                x_end - x_offset):
                        continue
                    masks.append(torch.as_tensor(component_mask,
                                                 dtype=torch.uint8))
                    labels.append(i)

        boxes = []
        valid_masks = []
        valid_labels = []
        for mask, label in zip(masks, labels):
            pos = torch.nonzero(mask)
            if pos.numel() == 0:
                continue
            else:
                xmin = torch.min(pos[:, 1])
                xmax = torch.max(pos[:, 1])
                ymin = torch.min(pos[:, 0])
                ymax = torch.max(pos[:, 0])
                if xmin < xmax and ymin < ymax:
                    boxes.append(torch.tensor(
                        [xmin, ymin, xmax, ymax],
                        dtype=torch.float32)
                        )
                    valid_masks.append(mask)
                    valid_labels.append(label)

        if len(valid_masks) == 0:
            valid_masks = [torch.zeros((img.height,
                                        img.width),
                           dtype=torch.uint8)]
            valid_labels = [1]
            boxes = [torch.tensor([0, 0, 1, 1],
                                  dtype=torch.float32)]

        masks = valid_masks
        labels = valid_labels
        boxes = torch.stack(boxes) if \
            boxes else torch.tensor([[0, 0, 1, 1]],
                                    dtype=torch.float32)

        img, masks, boxes, _ = pad_image_to_size(img,
                                                 masks,
                                                 boxes)

        masks = torch.stack(masks) if \
            masks else torch.zeros((1,
                                    PADDING_VALUE,
                                    PADDING_VALUE), dtype=torch.uint8)
        target = {
            'boxes': boxes,
            'labels': torch.as_tensor(labels, dtype=torch.int64),
            'masks': masks,
            'image_id': torch.tensor([image_idx]),
            'area': (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]),
            'iscrowd': torch.zeros((len(labels),), dtype=torch.int64),
        }

        if self.transforms:
            img = self.transforms(img)

        return img, target

    def __len__(self):
        return len(self.samples)


def collate_fn(batch):
    return tuple(zip(*batch))


def get_dataloader(train_folders,
                   valid_folders,
                   batch_size=4,
                   num_workers=4):
    train_transform = v2.Compose([
        v2.ToImage(),
        v2.ConvertImageDtype(torch.float)
    ])
    valid_transform = v2.Compose([
        v2.ToImage(),
        v2.ConvertImageDtype(torch.float)
    ])

    train_dataset = SegmentationDataset(train_folders,
                                        transforms=train_transform)
    valid_dataset = SegmentationDataset(valid_folders,
                                        transforms=valid_transform)

    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              collate_fn=collate_fn,
                              num_workers=num_workers)
    valid_loader = DataLoader(valid_dataset,
                              batch_size=batch_size,
                              shuffle=False,
                              collate_fn=collate_fn,
                              num_workers=num_workers)

    return train_loader, valid_loader


class TestDataset(Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        self.image_paths = [os.path.join(root, f) for
                            f in os.listdir(root) if
                            f.endswith('.tif')]

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("RGB")
        filename = os.path.basename(img_path)

        sub_images, offsets, orig_size = split_image_into_patches(img)

        processed_images = []
        processed_orig_sizes = []
        for sub_img in sub_images:
            padded_img, _, _, sub_orig_size = pad_image_to_size(sub_img)
            if self.transforms:
                padded_img = self.transforms(padded_img)
            processed_images.append(padded_img)
            processed_orig_sizes.append(sub_orig_size)

        return (processed_images,
                filename,
                orig_size,
                offsets,
                processed_orig_sizes)

    def __len__(self):
        return len(self.image_paths)


def get_test_dataloader(root, batch_size=4, num_workers=4):
    test_transform = v2.Compose([
        v2.ToImage(),
        v2.ConvertImageDtype(torch.float)
    ])

    test_dataset = TestDataset(root, transforms=test_transform)
    test_loader = DataLoader(test_dataset,
                             batch_size=batch_size,
                             shuffle=False,
                             collate_fn=collate_fn,
                             num_workers=num_workers)

    return test_loader
