from time import sleep
from utils.model import create_model
from utils.training_utils import (SaveBestModel,
                                  train_one_epoch,
                                  val_one_epoch,
                                  get_datasets)
import torch
import os
import time
from dataclasses import dataclass
from simple_parsing import ArgumentParser
from eval import evaluate_model
from torch.utils.tensorboard import SummaryWriter


def train(
    train_dataset,
    val_dataset,
    epochs=2,
    batch_size=8,
    exp_folder="exp",
    val_eval_freq=1,
):

    date_format = "%d-%m-%Y-%H-%M-%S"
    date_string = time.strftime(date_format)

    exp_folder = os.path.join("exp", "summary", date_string)
    writer = SummaryWriter(exp_folder)

    def custom_collate(data):
        return data

    # Dataloaders --
    train_dl = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=custom_collate,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    val_dl = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=custom_collate,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    # Device --
    device = torch.device("cuda:0")\
        if torch.cuda.is_available() else torch.device("cpu")

    # Model --
    faster_rcnn_model = create_model(
                                    train_dataset.get_total_classes_count() + 1
                                    )
    faster_rcnn_model = faster_rcnn_model.to(device)

    backbone_params = []
    # fpn_params = []
    rpn_params = []
    roi_params = []

    for name, param in faster_rcnn_model.named_parameters():
        if 'backbone' in name:
            backbone_params.append(param)
        elif 'rpn' in name:
            rpn_params.append(param)
        elif 'roi_heads' in name:
            roi_params.append(param)
        else:
            pass

    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': 0.00003},   # backbone 和 FPN
        {'params': rpn_params, 'lr': 0.0003},         # RPN
        {'params': roi_params, 'lr': 0.0003},         # ROI Head
    ])

    num_epochs = epochs
    save_best_model = SaveBestModel(output_dir=exp_folder)

    for epoch in range(num_epochs):

        faster_rcnn_model, optimizer, writer, epoch_loss = train_one_epoch(
            faster_rcnn_model,
            train_dl,
            optimizer,
            writer,
            epoch + 1,
            num_epochs,
            device,
        )

        sleep(0.1)

        writer, val_epoch_loss = val_one_epoch(
            faster_rcnn_model,
            val_dl,
            writer,
            epoch + 1,
            num_epochs,
            device,
            log=True,
        )

        sleep(0.1)

        current_weight_path = os.path.join(exp_folder,
                                           f"current_model\
                                            _epoch_{epoch+1}.pth")
        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": faster_rcnn_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }, current_weight_path)

        eval_result = evaluate_model(image_dir=val_dataset.image_folder,
                                     gt_ann_file=val_dataset.annotations_file,
                                     model_weight=current_weight_path)

        sleep(0.1)

        writer.add_scalar("Val/AP_50_95", eval_result['AP_50_95'], epoch + 1)
        writer.add_scalar("Val/AP_50", eval_result['AP_50'], epoch + 1)

        sleep(0.1)

        save_best_model(eval_result['AP_50_95'],
                        epoch,
                        faster_rcnn_model,
                        optimizer)

    writer.add_hparams(
        {"epochs": epochs, "batch_size": batch_size},
        {"Train/total_loss": epoch_loss, "Val/total_loss": val_epoch_loss},
    )


@dataclass
class DatasetPaths:
    train_image_dir: str = "./dataset/nycu-hw2-data/train"
    val_image_dir: str = "./dataset/nycu-hw2-data/valid"
    train_coco_json: str = "./dataset/nycu-hw2-data/train.json"
    val_coco_json: str = "./dataset/nycu-hw2-data/valid.json"


@dataclass
class TrainingConfig:
    epochs: int = 50
    batch_size: int = 4
    val_eval_freq: int = 5
    exp_folder: str = 'exp'


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_arguments(DatasetPaths, dest='dataset_config')
    parser.add_arguments(TrainingConfig, dest='training_config')
    args = parser.parse_args()

    dataset_config: DatasetPaths = args.dataset_config
    training_config: TrainingConfig = args.training_config

    (train_ds,
     val_ds) = get_datasets(train_image_dir=dataset_config.train_image_dir,
                            train_coco_json=dataset_config.train_coco_json,
                            val_image_dir=dataset_config.val_image_dir,
                            val_coco_json=dataset_config.val_coco_json)
    train(train_ds, val_ds,
          epochs=training_config.epochs,
          batch_size=training_config.batch_size,
          val_eval_freq=training_config.val_eval_freq,
          exp_folder=training_config.exp_folder)
