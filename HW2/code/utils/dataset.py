from collections import namedtuple
import abc
import cv2
import copy
import torch
import os
import json
from torch.utils.data import Dataset
import numpy as np
from collections import defaultdict
from torchvision import ops
import matplotlib.patches as patches
from torchvision import transforms as T
from PIL import Image

COCOBox_base = namedtuple("COCOBox", ["xmin", "ymin", "width", "height"])
VOCBox_base = namedtuple("VOCBox", ["xmin", "ymin", "xmax", "ymax"])


class COCOBox(COCOBox_base):
    def area(self):
        return self.width * self.height


class VOCBox(VOCBox_base):
    def area(self):
        return (self.xmax - self.xmin) * (self.ymax - self.ymin)


# Define the abstract base class for loading datasets
class DatasetLoader(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def load_images(self):
        pass

    @abc.abstractmethod
    def load_annotations(self):
        pass


# the dataset class
class CocoDataset(Dataset):
    def __init__(self, image_folder,
                 annotations_file,
                 width,
                 height,
                 transforms=None):

        self.transforms = transforms
        self.image_folder = image_folder
        self.annotations_file = annotations_file
        self.height = height
        self.width = width

        if not isinstance(self.image_folder, str):
            raise ValueError("image_folder should be a string")

        if not isinstance(annotations_file, str):
            raise ValueError("annotations_file should be a string")

        self.annotations_file = annotations_file
        self.image_folder = image_folder
        self.width = width
        self.height = height

        with open(annotations_file, "r") as f:
            self.annotations = json.load(f)

        self.image_ids = defaultdict(list)
        for i in self.annotations["images"]:
            self.image_ids[i["id"]] = i

        self.annotation_ids = defaultdict(list)
        for i in self.annotations["annotations"]:
            self.annotation_ids[i["image_id"]].append(i)

        self.cats_id2label = {}
        self.label_names = []

        first_label_id = self.annotations["categories"][0]["id"]
        if first_label_id == 0:
            for i in self.annotations["categories"][1:]:
                self.cats_id2label[i["id"]] = i["name"]
                self.label_names.append(i["name"])
        if first_label_id == 1:
            for i in self.annotations["categories"]:
                self.cats_id2label[i["id"]] = i["name"]
                self.label_names.append(i["name"])
        if first_label_id > 1:
            raise AssertionError(
                "Something went wrong in categories, \
                    check the annotation file!"
            )

    def get_total_classes_count(self):
        return len(self.cats_id2label)

    def get_classnames(self):
        return [v for k, v in self.cats_id2label.items()]

    def load_images_annotations(self, index):
        index += 1  # 1-based index
        image_info = self.image_ids[index]
        if index not in self.image_ids:
            print("Image not found in the dataset")
            print(index)
            print(image_info)
            return None
        if index == 0:
            print("No image_info")
            print(index)
            print(image_info)
            return None
        image_path = os.path.join(self.image_folder, image_info["file_name"])

        image = cv2.imread(image_path)
        rimage = cv2.cvtColor(
            image, cv2.COLOR_BGR2RGB
        )

        rimage = Image.fromarray(rimage)

        image_height, image_width = (
            image_info["height"],
            image_info["width"],
        )
        anno_info = self.annotation_ids[index]

        if len(anno_info) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0, 1), dtype=torch.int64)
            iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        else:
            boxes = []
            labels_id = []

            for ainfo in anno_info:
                xmin, ymin, w, h = ainfo["bbox"]
                xmax, ymax = xmin + w, ymin + h

                xmin_final = xmin
                xmax_final = xmax
                ymin_final = ymin
                ymax_final = ymax

                category_id = ainfo["category_id"]

                boxes.append([xmin_final, ymin_final, xmax_final, ymax_final])
                labels_id.append(category_id)

            boxes = torch.as_tensor(
                boxes, dtype=torch.float32
            )  # bounding box to tensor
            area = (boxes[:, 3] - boxes[:, 1]) * (
                boxes[:, 2] - boxes[:, 0]
            )  # area of the bounding boxes
            iscrowd = torch.zeros(
                (boxes.shape[0],), dtype=torch.int64
            )  # no crowd instances
            labels = torch.as_tensor(labels_id, dtype=torch.int64)

        # final `target` dictionary
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["area"] = area
        target["iscrowd"] = iscrowd
        image_id = torch.tensor([index])
        target["image_id"] = image_id

        return {
            "image": rimage,
            "height": image_height,
            "width": image_width,
            "target": target,
        }

    @staticmethod
    def transform_image_for_inference(image_path, width, height):

        image = cv2.imread(image_path)
        ori_h, ori_w, _ = image.shape

        oimage = copy.deepcopy(image)
        oimage = Image.fromarray(oimage)
        oimage = T.ToTensor()(oimage)

        rimage = cv2.cvtColor(
            image, cv2.COLOR_BGR2RGB
        )

        rimage = Image.fromarray(rimage)
        rimage = T.ToTensor()(rimage)

        transform_info = {'original_width': ori_w,
                          'original_height': ori_h,
                          'resized_width': ori_w,
                          'resized_height': ori_h,
                          'resized_image': rimage,
                          'original_image': oimage}

        return transform_info

    @staticmethod
    def display_bbox(
        bboxes,
        fig,
        ax,
        classes=None,
        in_format="xyxy",
        color="y",
        line_width=3
    ):
        if type(bboxes) is np.ndarray:
            bboxes = torch.from_numpy(bboxes)
        if classes:
            assert len(bboxes) == len(classes)
        # convert boxes to xywh format
        bboxes = ops.box_convert(bboxes, in_fmt=in_format, out_fmt="xywh")
        c = 0
        for box in bboxes:
            x, y, w, h = box.numpy()
            # display bounding box
            rect = patches.Rectangle(
                (x, y),
                w,
                h,
                linewidth=line_width,
                edgecolor=color,
                facecolor="none"
            )
            ax.add_patch(rect)
            # display category
            if classes:
                if classes[c] == "pad":
                    continue
                ax.text(
                    x + 5,
                    y + 20,
                    classes[c],
                    bbox=dict(facecolor="yellow",
                              alpha=0.5)
                )
            c += 1

        return fig, ax

    def __getitem__(self, idx):

        sample = self.load_images_annotations(idx)
        image = sample["image"]  # PIL image
        target = sample["target"]

        image_np = np.array(image)

        if self.transforms:

            boxes = target["boxes"].numpy().tolist()
            labels = target["labels"].numpy().tolist()

            transformed = self.transforms(image=image_np,
                                          bboxes=boxes,
                                          labels=labels)
            image_np = transformed["image"]
            boxes = transformed["bboxes"]
            labels = transformed["labels"]

            target["boxes"] = torch.as_tensor(boxes,
                                              dtype=torch.float32
                                              ).reshape(-1, 4)
            target["labels"] = torch.as_tensor(labels, dtype=torch.int64)

        image_tensor = T.ToTensor()(image_np)
        return image_tensor, target

    def __len__(self):
        return len(self.image_ids)
