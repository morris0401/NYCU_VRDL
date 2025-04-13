import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_V2_Weights,
)
from torchvision.models.detection.anchor_utils import AnchorGenerator
import warnings
import torch
from torchvision.models.resnet import Bottleneck
from torch import nn
warnings.filterwarnings("ignore", category=UserWarning)


def create_stochastic_depth_hook(p: float):
    """
    創建一個 Forward Hook，使用 torchvision.ops.stochastic_depth。
    """
    def hook(module: nn.Module,
             input: torch.Tensor,
             output: torch.Tensor) -> torch.Tensor:
        """實際的 Hook 函數。"""
        return torchvision.ops.stochastic_depth(output,
                                                p,
                                                mode="row",
                                                training=module.training)
    return hook


def add_stochastic_depth_to_resnet(model_resnet_part: nn.Module,
                                   stoch_depth_prob: float) -> nn.Module:
    """
    將線性隨機深度添加到 ResNet 模型或其一部分 (例如 FasterRCNN 的 backbone.body)。
    """
    bottlenecks = [m for m in model_resnet_part.modules()
                   if isinstance(m, Bottleneck)]
    total_blocks = len(bottlenecks)

    if total_blocks == 0:
        print(f"警告：在 {model_resnet_part.__class__.__name__} \
              中未找到 Bottleneck 區塊。")
        return model_resnet_part

    for i, bottleneck_module in enumerate(bottlenecks):
        drop_p = (i / (total_blocks - 1)) * stoch_depth_prob \
            if total_blocks > 1 else 0.0
        bottleneck_module.bn3.register_forward_hook(
            create_stochastic_depth_hook(drop_p)
            )

    print(f"成功為 {model_resnet_part.__class__.__name__} 中的 {total_blocks} 個 \
          Bottleneck 區塊添加了線性隨機深度 (最終捨棄機率: {stoch_depth_prob:.4f})。")
    return model_resnet_part


def create_model(num_classes, checkpoint=None, device='cpu'):
    """
    Create a model for object detection using the Faster R-CNN architecture.

    Parameters:
    - num_classes (int): The number of classes for object detection.
        (Total classes + 1 [for background class])
    - checkpoint (str): checkpoint path for the pretrained custom model
    - device (str): cpu / cuda
    Returns:
    - model (torchvision.models.detection.fasterrcnn_resnet50_fpn_v2):
        The created model for object detection.
    """
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(
        weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT,
        pretrained_backbone=True,
    )

    anchor_sizes = ((16, 32,), (32, 64,), (64, 128,), (128, 256,), (256, 512,))
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)

    anchor_generator = AnchorGenerator(sizes=anchor_sizes,
                                       aspect_ratios=aspect_ratios)

    model.rpn.anchor_generator = anchor_generator

    num_anchors_per_location = len(anchor_sizes[0]) * len(aspect_ratios[0])

    in_channels = model.rpn.head.conv[0][0].in_channels

    model.rpn.head = torchvision.models.detection.rpn.RPNHead(
        in_channels=in_channels,
        num_anchors=num_anchors_per_location,
    )

    # Add stochastic depth to the backbone
    final_drop_prob = 0.5
    model.backbone.body = add_stochastic_depth_to_resnet(model.backbone.body,
                                                         final_drop_prob)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    if checkpoint:
        checkpoint = torch.load(checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model = model.to(device)

    return model


if __name__ == "__main__":
    model = create_model(num_classes=11)
    print(model.rpn.anchor_generator.sizes)
