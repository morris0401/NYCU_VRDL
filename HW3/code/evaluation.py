import torch
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils
import numpy as np


def create_coco_gt(dataset, device):
    coco_gt = {
        "images": [],
        "annotations": [],
        "categories": [{"id": i, "name": f"class{i}"}
                       for i in range(1, 5)]
    }
    ann_id = 1
    for idx in range(len(dataset)):
        img, target = dataset[idx]
        img_id = int(target['image_id'][0])
        coco_gt["images"].append({
            "id": img_id,
            "width": img.shape[-1],
            "height": img.shape[-2],
            "file_name": f"image_{img_id}.tif"
        })

        boxes = target['boxes'].cpu().numpy()
        labels = target['labels'].cpu().numpy()
        masks = target['masks'].cpu().numpy()
        areas = target['area'].cpu().numpy()
        iscrowd = target['iscrowd'].cpu().numpy()

        for box, label, mask, area, crowd in zip(boxes,
                                                 labels,
                                                 masks,
                                                 areas,
                                                 iscrowd):
            if label == 0:
                continue
            x, y, x2, y2 = box
            w, h = x2 - x, y2 - y
            if w <= 0 or h <= 0:
                continue
            rle = maskUtils.encode(np.asfortranarray(mask.astype(np.uint8)))
            rle['counts'] = rle['counts'].decode('utf-8')
            coco_gt["annotations"].append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": int(label),
                "bbox": [float(x), float(y), float(w), float(h)],
                "area": float(area),
                "segmentation": rle,
                "iscrowd": int(crowd)
            })
            ann_id += 1

    return coco_gt


def create_coco_dt(outputs, image_ids, device):
    coco_dt = []
    ann_id = 1
    for output, img_id in zip(outputs, image_ids):
        boxes = output['boxes'].cpu().numpy()
        scores = output['scores'].cpu().numpy()
        labels = output['labels'].cpu().numpy()
        masks = output['masks'].cpu().numpy()

        for box, score, label, mask in zip(boxes, scores, labels, masks):
            if label == 0 or score < 0.05:
                continue
            x, y, x2, y2 = box
            w, h = x2 - x, y2 - y
            if w <= 0 or h <= 0:
                continue
            mask = (mask.squeeze() > 0.5).astype(np.uint8)
            rle = maskUtils.encode(np.asfortranarray(mask))
            rle['counts'] = rle['counts'].decode('utf-8')
            coco_dt.append({
                "id": ann_id,
                "image_id": int(img_id),
                "category_id": int(label),
                "bbox": [float(x), float(y), float(w), float(h)],
                "score": float(score),
                "segmentation": rle
            })
            ann_id += 1

    return coco_dt


def validate(model, data_loader, device):
    model.eval()
    coco_gt = create_coco_gt(data_loader.dataset, device)
    coco = COCO()
    coco.dataset = coco_gt
    coco.createIndex()

    coco_dt = []
    image_ids = []
    with torch.no_grad():
        for images, targets in tqdm(data_loader,
                                    desc="Validating",
                                    leave=False):
            images = list(img.to(device) for img in images)
            targets = [{k: v.to(device) for k, v in t.items()}
                       for t in targets]
            outputs = model(images)
            image_ids.extend([t['image_id'].item() for t in targets])
            coco_dt.extend(create_coco_dt(outputs,
                                          [t['image_id'].item()
                                           for t in targets],
                                          device))

    if not coco_dt:
        return 0.0

    coco_dt_coco = coco.loadRes(coco_dt)
    coco_eval = COCOeval(coco, coco_dt_coco, 'segm')

    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    mAP = coco_eval.stats[0] if coco_eval.stats[0] >= 0 else 0.0
    return mAP
