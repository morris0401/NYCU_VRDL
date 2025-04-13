import os
import json
import csv
import torch
from tqdm import tqdm
import gc
from utils.model_utils import InferFasterRCNN
from utils.dataset import CocoDataset


def save_json(data, file_path):
    with open(file_path, 'w') as file:
        json.dump(data, file)


def save_csv(data, file_path):
    with open(file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["image_id", "pred_label"])
        for row in data:
            writer.writerow(row)


def evaluate_model(image_dir, model_weight):
    num_classes = 11
    classnames = [str(i) for i in range(10)]

    device = torch.device("cuda:2") \
        if torch.cuda.is_available() \
        else torch.device("cpu")

    IF_C = InferFasterRCNN(num_classes=num_classes,
                           classnames=classnames)
    IF_C.load_model(checkpoint=model_weight,
                    device=device)

    image_files = sorted([f for f in os.listdir(image_dir)
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

    json_results = []
    csv_results = []

    for file_name in tqdm(image_files, total=len(image_files)):
        image_id = int(file_name.split('.')[0])
        image_path = os.path.join(image_dir, file_name)
        transform_info = CocoDataset.transform_image_for_inference(image_path,
                                                                   width=640,
                                                                   height=640)
        result = IF_C.infer_image(transform_info=transform_info,
                                  visualize=False)

        if "unscaled_boxes" in result and len(result["unscaled_boxes"]) > 0:
            pred_boxes_xyxy = result['unscaled_boxes']
            pred_boxes_xywh = [[box[0], box[1], box[2]-box[0], box[3]-box[1]]
                               for box in pred_boxes_xyxy]
            pred_scores = result['scores']
            pred_labels = result['labels']

            pred_labels = [int(label) for label in pred_labels]

            for bbox, score, label in zip(pred_boxes_xywh,
                                          pred_scores,
                                          pred_labels):
                json_results.append({
                    "image_id": image_id,
                    "bbox": bbox,
                    "score": float(score),
                    "category_id": int(label)
                })

            pred_labels_1 = [label
                             for i, label in enumerate(pred_labels)
                             if pred_scores[i] >= 0.80]
            if len(pred_labels_1) == 0:
                pred_labels_2 = [label
                                 for i, label in enumerate(pred_labels)
                                 if pred_scores[i] >= 0.60]
                if len(pred_labels_2) == 0:
                    pred_labels_3 = [label
                                     for i, label in enumerate(pred_labels)
                                     if pred_scores[i] >= 0.30]
                    if len(pred_labels_3) == 0:
                        csv_results.append([image_id, "-1"])
                        continue
                    else:
                        pred_labels = pred_labels_3
                else:
                    pred_labels = pred_labels_2
            else:
                pred_labels = pred_labels_1

            detections = list(zip(result['unscaled_boxes'], pred_labels))
            detections_sorted = sorted(detections, key=lambda x: x[0][0])
            pred_digit_str = "".join([str(label - 1) for _, label
                                      in detections_sorted])

            if len(pred_digit_str) >= 9:
                pred_digit_str = pred_digit_str[:9]

            csv_results.append([image_id, int(pred_digit_str)])
        else:
            csv_results.append([image_id, "-1"])

    json_save_path = 'result8.json'
    csv_save_path = 'result8.csv'

    save_json(json_results, json_save_path)
    save_csv(csv_results, csv_save_path)

    del IF_C
    torch.cuda.empty_cache()
    gc.collect()

    print(f"Saved JSON results to {json_save_path}")
    print(f"Saved CSV results to {csv_save_path}")


if __name__ == "__main__":
    image_directory = "./dataset/nycu-hw2-data/test"
    model_weights = "./models/excellent.pth"

    evaluate_model(image_directory, model_weights)

# 10 470, 12 468, 13 467, 18 466,
# 20 469, 22 469, 27 470, 30 468,
# 33 468, 35 473, 36 469, 38 466, 39 466

# 12 good 0.798 085445, 085399
# 18 private task1 good
