import torch
import cv2
import os
from .model import create_model
import numpy as np
import matplotlib.pyplot as plt


class InferFasterRCNN:
    def __init__(self, num_classes=None, classnames=[]):

        assert type(num_classes) is not type(None), "Define number of classes"

        self.num_classes = num_classes  # total_class_no + 1 (for background)

        self.classnames = ["__background__"]
        self.classnames.extend(classnames)

        self.colors = np.random.uniform(0, 255, size=(len(self.classnames), 3))

        assert (
            len(self.classnames) == self.num_classes
        ), f"num_classes: {self.num_classes}, \
            len(classnames): {len(self.classnames)}.\
            num_classes should be equal to count of \
                actual classes in classnames list without background + 1"

    def load_model(self, checkpoint, device="cpu"):
        self.device = device
        self.model = create_model(
            self.num_classes, checkpoint=checkpoint, device=self.device
        )
        self.model = self.model.eval()

    def infer_image(self,
                    transform_info,
                    detection_threshold=0.00,
                    visualize=False):

        '''
        image : original unscaled image
        '''

        display_unscaled = True
        h_ratio = transform_info['original_height'] \
            / transform_info['resized_height']
        w_ratio = transform_info['original_width'] \
            / transform_info['resized_width']

        orig_image = transform_info['resized_image']
        orig_image = orig_image.cpu().numpy()
        orig_image = np.transpose(orig_image, (1, 2, 0))
        orig_image = np.ascontiguousarray(orig_image, dtype=np.float32)
        image = torch.unsqueeze(transform_info['resized_image'], 0)

        with torch.no_grad():
            self.model = self.model.to(self.device)
            outputs = self.model(image.to(self.device))

        # load all detection to CPU for further operations
        outputs = [{k: v.to("cpu") for k, v in t.items()} for t in outputs]

        results = {}
        _f_boxes, _f_scores, _f_labels = [], [], []

        # carry further only if there are detected boxes
        if len(outputs[0]["boxes"]) != 0:
            boxes = outputs[0]["boxes"].data.numpy()  # xyxy
            scores = outputs[0]["scores"].data.numpy()
            labels = outputs[0]["labels"].cpu().numpy()

            # filter out boxes according to `detection_threshold`
            for i in range(len(boxes)):
                if scores[i] >= detection_threshold:
                    _f_boxes.append(boxes[i])
                    _f_labels.append(labels[i])
                    _f_scores.append(scores[i])

            boxes, labels, scores = _f_boxes, _f_labels, _f_scores

            draw_boxes = boxes.copy()

            # get all the predicited class names
            pred_classes = [
                self.classnames[i] for i in labels
            ]

            results['unscaled_boxes'] = [[i[0]*w_ratio,
                                          i[1]*h_ratio,
                                          i[2]*w_ratio,
                                          i[3]*h_ratio] for i in boxes]
            results['scaled_boxes'] = boxes
            results['scores'] = scores
            results['pred_classes'] = pred_classes
            results['labels'] = labels

            if not display_unscaled:
                for j, box in enumerate(draw_boxes):
                    class_name = pred_classes[j]
                    color = self.colors[self.classnames.index(class_name)]
                    cv2.rectangle(
                        orig_image,
                        (int(box[0]), int(box[1])),
                        (int(box[2]), int(box[3])),
                        color,
                        2,
                    )
                    cv2.putText(
                        orig_image,
                        class_name,
                        (int(box[0]), int(box[1] - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        color,
                        2,
                        lineType=cv2.LINE_AA,
                    )

                if visualize:
                    plt.figure(figsize=(10, 10))
                    plt.imshow(orig_image[:, :, ::-1])
                    plt.show()

            else:
                # draw the bounding boxes and write the class name on top of it
                draw_boxes_scaled = results['unscaled_boxes']
                scaled_orig_image = transform_info['original_image']
                scaled_orig_image = scaled_orig_image.cpu().numpy()
                scaled_orig_image = np.transpose(scaled_orig_image, (1, 2, 0))
                scaled_orig_image = np.ascontiguousarray(scaled_orig_image,
                                                         dtype=np.float32
                                                         )

                for j, box in enumerate(draw_boxes_scaled):
                    class_name = pred_classes[j]
                    color = self.colors[self.classnames.index(class_name)]
                    cv2.rectangle(
                        scaled_orig_image,
                        (int(box[0]), int(box[1])),
                        (int(box[2]), int(box[3])),
                        color,
                        2,
                    )
                    cv2.putText(
                        scaled_orig_image,
                        class_name,
                        (int(box[0]), int(box[1] - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        color,
                        2,
                        lineType=cv2.LINE_AA,
                    )

                if visualize:
                    plt.figure(figsize=(10, 10))
                    plt.imshow(scaled_orig_image)  # [:,:,::-1])
                    plt.show()

        return results

    def infer_image_path(self,
                         image_path,
                         detection_threshold=0.5,
                         visualize=False):

        image = cv2.imread(image_path)
        orig_image = image.copy()

        # BGR to RGB
        image = cv2.cvtColor(orig_image,
                             cv2.COLOR_BGR2RGB).astype(np.float32)
        # make the pixel range between 0 and 1
        image /= 255.0
        # bring color channels to front
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        # convert to tensor
        image = torch.tensor(image, dtype=torch.float).cpu()

        # add batch dimension
        image = torch.unsqueeze(image, 0)
        with torch.no_grad():
            self.model = self.model.to(self.device)
            outputs = self.model(image.to(self.device))

        # load all detection to CPU for further operations
        outputs = [{k: v.to("cpu") for k, v in t.items()} for t in outputs]

        # carry further only if there are detected boxes
        if len(outputs[0]["boxes"]) != 0:
            boxes = outputs[0]["boxes"].data.numpy()
            scores = outputs[0]["scores"].data.numpy()

            # filter out boxes according to `detection_threshold`
            boxes = boxes[scores >= detection_threshold].astype(np.int32)
            draw_boxes = boxes.copy()

            # get all the predicited class names
            pred_classes = [
                self.classnames[i] for i in outputs[0]["labels"].cpu().numpy()
            ]

            # draw the bounding boxes and write the class name on top of it
            for j, box in enumerate(draw_boxes):
                class_name = pred_classes[j]
                color = self.colors[self.classnames.index(class_name)]
                cv2.rectangle(
                    orig_image,
                    (int(box[0]), int(box[1])),
                    (int(box[2]), int(box[3])),
                    color,
                    2,
                )
                cv2.putText(
                    orig_image,
                    class_name,
                    (int(box[0]), int(box[1] - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2,
                    lineType=cv2.LINE_AA,
                )

            if visualize:
                plt.figure(figsize=(10, 10))
                plt.imshow(orig_image[:, :, ::-1])
                plt.show()

        return outputs


class SaveBestModel:
    """
    Class to save the best model while training. If the current epoch's
    validation loss is less than the previous least less, then save the
    model state.
    """

    def __init__(  # Modify to MAP
        self, best_map=0.0,
        output_dir="weight_outputs",
    ):
        self.best_map = best_map

        os.makedirs(output_dir, exist_ok=True)

        self.output_dir = output_dir

    def __call__(self, current_map, epoch, model, optimizer):
        if current_map > self.best_map:
            self.best_map = current_map
            print(f"\nBest MAP: {self.best_map}")
            print(f"\nSaving best model for epoch: {epoch+1}\n")
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                f"{self.output_dir}/best_model.pth",
            )
