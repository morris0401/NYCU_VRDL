import json
import numpy as np
import os
import torch
import torchvision
from tqdm import tqdm
import pycocotools.mask
from data import get_test_dataloader
from model import MaskRCNNModel
from train import set_seed
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

NAMING = "src_a12"
PADDING_VALUE = 400


def remove_padding_from_predictions(boxes,
                                    masks,
                                    orig_size,
                                    padded_size=(PADDING_VALUE,
                                                 PADDING_VALUE)
                                    ):
    orig_height, orig_width = orig_size
    padded_height, padded_width = padded_size

    pad_left = (padded_width - orig_width) // 2 \
        if orig_width < padded_width else 0
    pad_top = (padded_height - orig_height) // 2 \
        if orig_height < padded_height else 0

    adjusted_boxes = boxes.copy()
    adjusted_boxes[:, 0] = np.maximum(boxes[:, 0] - pad_left,
                                      0)  # xmin
    adjusted_boxes[:, 2] = np.minimum(boxes[:, 2] - pad_left,
                                      orig_width)  # xmax
    adjusted_boxes[:, 1] = np.maximum(boxes[:, 1] - pad_top,
                                      0)  # ymin
    adjusted_boxes[:, 3] = np.minimum(boxes[:, 3] - pad_top,
                                      orig_height)  # ymax

    adjusted_masks = masks.copy()
    if pad_left > 0 or pad_top > 0:
        adjusted_masks = adjusted_masks[:,
                                        :,
                                        pad_top:pad_top+orig_height,
                                        pad_left:pad_left+orig_width]

    return adjusted_boxes, adjusted_masks


def merge_predictions(predictions_list,
                      offsets,
                      orig_size,
                      iou_threshold=0.5):
    orig_height, orig_width = orig_size
    all_boxes = []
    all_scores = []
    all_labels = []
    all_masks = []

    for preds, (x_offset, y_offset) in zip(predictions_list,
                                           offsets):
        boxes = preds['boxes']
        scores = preds['scores']
        labels = preds['labels']
        masks = preds['masks']

        adjusted_boxes = boxes.copy()
        adjusted_boxes[:, 0] += x_offset  # xmin
        adjusted_boxes[:, 2] += x_offset  # xmax
        adjusted_boxes[:, 1] += y_offset  # ymin
        adjusted_boxes[:, 3] += y_offset  # ymax

        adjusted_masks = np.zeros((masks.shape[0],
                                   1,
                                   orig_height,
                                   orig_width), dtype=masks.dtype)
        for i, mask in enumerate(masks):
            mask = mask[0]  # [1, H, W] -> [H, W]
            mask_h, mask_w = mask.shape
            x_start = int(x_offset)
            y_start = int(y_offset)
            x_end = min(x_start + mask_w, orig_width)
            y_end = min(y_start + mask_h, orig_height)
            adjusted_masks[i, 0, y_start:y_end, x_start:x_end] = \
                mask[:y_end-y_start, :x_end-x_start]

        all_boxes.append(adjusted_boxes)
        all_scores.append(scores)
        all_labels.append(labels)
        all_masks.append(adjusted_masks)

    all_boxes = np.concatenate(all_boxes, axis=0) \
        if all_boxes else np.empty((0, 4))
    all_scores = np.concatenate(all_scores, axis=0) \
        if all_scores else np.empty((0,))
    all_labels = np.concatenate(all_labels, axis=0) \
        if all_labels else np.empty((0,), dtype=np.int64)
    all_masks = np.concatenate(all_masks, axis=0) \
        if all_masks else np.empty((0, 1, orig_height, orig_width))

    if len(all_boxes) > 0:
        keep = torchvision.ops.nms(
            torch.from_numpy(all_boxes).float(),
            torch.from_numpy(all_scores).float(),
            iou_threshold=iou_threshold
        ).numpy()
        all_boxes = all_boxes[keep]
        all_scores = all_scores[keep]
        all_labels = all_labels[keep]
        all_masks = all_masks[keep]

    return {
        'boxes': all_boxes,
        'scores': all_scores,
        'labels': all_labels,
        'masks': all_masks
    }


def main():
    seed = 77
    set_seed(seed=seed)

    device = torch.device('cuda'
                          if torch.cuda.is_available()
                          else 'cpu')

    with open('../dataset/test_image_name_to_ids.json', 'r') as f:
        image_name_to_ids = json.load(f)
    name_to_id_map = {os.path.basename(item['file_name']):
                      item['id'] for item in image_name_to_ids}

    test_root = '../dataset/test_release'
    test_loader = get_test_dataloader(
        root=test_root,
        batch_size=1,
        num_workers=2,
    )

    num_classes = 5
    model = MaskRCNNModel(num_classes=num_classes,
                          pretrained=True).to(device)
    checkpoint = f'../models/{NAMING}_seed{seed}.pth'
    model.load_state_dict(torch.load(checkpoint,
                                     map_location=device))
    model.eval()

    predictions = []

    with torch.no_grad():
        for images, filenames, orig_sizes, offsets, sub_orig_sizes \
                in tqdm(test_loader, desc="Testing", leave=False):
            filename = filenames[0]
            orig_size = orig_sizes[0]
            offset_list = offsets[0]
            sub_orig_size_list = sub_orig_sizes[0]
            image_id = name_to_id_map.get(filename)
            if image_id is None:
                print(f"Warning: No image_id \
                      found for filename {filename}")
                continue
            sub_images = images[0]
            sub_predictions = []
            for img, sub_orig_size in zip(sub_images,
                                          sub_orig_size_list):
                img = img.to(device)
                output = model([img])[0]

                boxes, masks = remove_padding_from_predictions(
                    output['boxes'].cpu().numpy(),
                    output['masks'].cpu().numpy(),
                    sub_orig_size
                )

                sub_predictions.append({
                    'boxes': boxes,
                    'scores': output['scores'].cpu().numpy(),
                    'labels': output['labels'].cpu().numpy(),
                    'masks': masks
                })

            merged_output = merge_predictions(sub_predictions,
                                              offset_list,
                                              orig_size,
                                              iou_threshold=0.5)

            boxes = merged_output['boxes']
            scores = merged_output['scores']
            labels = merged_output['labels']
            masks = merged_output['masks']

            for box, score, label, mask in zip(boxes,
                                               scores,
                                               labels,
                                               masks):
                if score < 0.05 or label == 0:
                    continue
                mask = mask[0]
                binary_mask = (mask > 0.5).astype(np.uint8)

                rle = pycocotools.mask.encode(
                    np.asfortranarray(binary_mask))
                rle['counts'] = rle['counts'].decode('utf-8')

                x1, y1, x2, y2 = box
                bbox = [float(x1),
                        float(y1),
                        float(x2 - x1),
                        float(y2 - y1)]

                predictions.append({
                    'image_id': int(image_id),
                    'bbox': bbox,
                    'score': float(score),
                    'category_id': int(label),
                    'segmentation': rle
                })

    output_file = 'test_predictions.json'
    with open(output_file, 'w') as f:
        json.dump(predictions, f, indent=2)

    print(f'Predictions saved to {output_file}')


if __name__ == "__main__":
    main()
