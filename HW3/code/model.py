import torchvision
from torchvision.models.detection.mask_rcnn import MaskRCNN, MaskRCNNPredictor
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.rpn import RPNHead
from torchvision.ops.feature_pyramid_network import (FeaturePyramidNetwork,
                                                     LastLevelMaxPool)
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
from torchvision.models import convnext_small, ConvNeXt_Small_Weights
from torchvision.models import convnext_base, ConvNeXt_Base_Weights
from torchvision.models import convnext_large, ConvNeXt_Large_Weights
from torchvision.models.feature_extraction import create_feature_extractor
import warnings
from torch import nn

warnings.filterwarnings("ignore", category=UserWarning)

# 定義 ConvNeXt 的輸入通道數
input_channels_dict = {
    "convnext_tiny": [96, 192, 384, 768],
    "convnext_small": [96, 192, 384, 768],
    "convnext_base": [128, 256, 512, 1024],
    "convnext_large": [192, 384, 768, 1536],
}


class BackboneWithFPN(nn.Module):
    def __init__(
        self,
        backbone,
        in_channels_list,
        out_channels,
        extra_blocks=None,
        norm_layer=None,
    ):
        super().__init__()

        if extra_blocks is None:
            extra_blocks = LastLevelMaxPool()

        self.body = backbone
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=in_channels_list,
            out_channels=out_channels,
            extra_blocks=extra_blocks,
            norm_layer=norm_layer,
        )
        self.out_channels = out_channels

    def forward(self, x):
        x = self.body(x)
        x = self.fpn(x)
        return x


def convnext_fpn_backbone(
    backbone_name="convnext_tiny",
    trainable_layers=3,
    extra_blocks=None,
    norm_layer=None,
    feature_dict={'1': '0', '3': '1', '5': '2', '7': '3'},
    out_channels=256,
    stochastic_depth_prob=0.5
):

    if backbone_name == "convnext_tiny":
        backbone = \
            convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT).features
        backbone = create_feature_extractor(backbone, feature_dict)
    elif backbone_name == "convnext_small":
        backbone = \
            convnext_small(weights=ConvNeXt_Small_Weights.DEFAULT).features
        backbone = create_feature_extractor(backbone, feature_dict)
    elif backbone_name == "convnext_base":
        backbone = \
            convnext_base(weights=ConvNeXt_Base_Weights.DEFAULT).features
        backbone = create_feature_extractor(backbone, feature_dict)
    elif backbone_name == "convnext_large":
        backbone = \
            convnext_large(weights=ConvNeXt_Large_Weights.DEFAULT).features
        backbone = create_feature_extractor(backbone, feature_dict)
    else:
        raise ValueError(f"Backbone names should be in \
                         {list(input_channels_dict.keys())}, \
                         got {backbone_name}")

    in_channels_list = input_channels_dict[backbone_name]

    if trainable_layers < 0 or trainable_layers > 8:
        raise ValueError(f"Trainable layers should be in \
                         the range [0,8], got {trainable_layers}")
    layers_to_train = ["7", "6", "5", "4", "3", "2", "1"][:trainable_layers]
    if trainable_layers == 8:
        layers_to_train.append("bn1")
    for name, parameter in backbone.named_parameters():
        if all([not name.startswith(layer) for layer in layers_to_train]):
            parameter.requires_grad_(False)

    if extra_blocks is None:
        extra_blocks = LastLevelMaxPool()

    return BackboneWithFPN(
        backbone, in_channels_list, out_channels,
        extra_blocks=extra_blocks, norm_layer=norm_layer
    )


class MaskRCNNModel(nn.Module):
    def __init__(self, num_classes=5, pretrained=True):
        super(MaskRCNNModel, self).__init__()

        # Initialize backbone with ConvNeXt + FPN
        backbone = convnext_fpn_backbone(
            backbone_name="convnext_base",
            trainable_layers=3 if pretrained else 8,
            out_channels=256
        )

        # Custom anchor generator
        anchor_sizes = ((16, 32,),
                        (32, 64,),
                        (64, 128,),
                        (128, 256,),
                        (256, 512,))
        aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
        anchor_generator = AnchorGenerator(sizes=anchor_sizes,
                                           aspect_ratios=aspect_ratios)

        # RPN
        rpn_head = RPNHead(
            in_channels=backbone.out_channels,
            num_anchors=len(anchor_sizes[0]) * len(aspect_ratios[0])
        )
        rpn_pre_nms_top_n = {"training": 2000, "testing": 1000}
        rpn_post_nms_top_n = {"training": 2000, "testing": 1000}
        rpn_nms_thresh = 0.7
        rpn_fg_iou_thresh = 0.7
        rpn_bg_iou_thresh = 0.3
        rpn_batch_size_per_image = 256
        rpn_positive_fraction = 0.5

        # ROI Heads
        box_roi_pool = torchvision.ops.MultiScaleRoIAlign(
            featmap_names=["0", "1", "2", "3"],
            output_size=7,
            sampling_ratio=2,
        )
        mask_roi_pool = torchvision.ops.MultiScaleRoIAlign(
            featmap_names=["0", "1", "2", "3"],
            output_size=14,
            sampling_ratio=2,
        )
        box_head = torchvision.models.detection.faster_rcnn.TwoMLPHead(
            in_channels=backbone.out_channels *
            box_roi_pool.output_size[0] ** 2,
            representation_size=1024,
        )
        box_predictor = FastRCNNPredictor(1024, num_classes)
        mask_head = torchvision.models.detection.mask_rcnn.MaskRCNNHeads(
            in_channels=backbone.out_channels,
            layers=(256, 256, 256, 256),
            dilation=1,
        )
        mask_predictor = MaskRCNNPredictor(
            in_channels=256,
            dim_reduced=256,
            num_classes=num_classes,
        )

        box_score_thresh = 0.05
        box_nms_thresh = 0.5
        box_detections_per_img = 200
        box_fg_iou_thresh = 0.5
        box_bg_iou_thresh = 0.5
        box_batch_size_per_image = 128
        box_positive_fraction = 0.25

        # Initialize Mask R-CNN
        self.model = MaskRCNN(
            backbone=backbone,
            num_classes=None,  # Set to None since mask_predictor is provided
            min_size=800,
            max_size=1333,
            rpn_anchor_generator=anchor_generator,
            rpn_head=rpn_head,
            rpn_pre_nms_top_n=rpn_pre_nms_top_n,
            rpn_post_nms_top_n=rpn_post_nms_top_n,
            rpn_nms_thresh=rpn_nms_thresh,
            rpn_fg_iou_thresh=rpn_fg_iou_thresh,
            rpn_bg_iou_thresh=rpn_bg_iou_thresh,
            rpn_batch_size_per_image=rpn_batch_size_per_image,
            rpn_positive_fraction=rpn_positive_fraction,
            box_roi_pool=box_roi_pool,
            box_head=box_head,
            box_predictor=box_predictor,
            box_score_thresh=box_score_thresh,
            box_nms_thresh=box_nms_thresh,
            box_detections_per_img=box_detections_per_img,
            box_fg_iou_thresh=box_fg_iou_thresh,
            box_bg_iou_thresh=box_bg_iou_thresh,
            box_batch_size_per_image=box_batch_size_per_image,
            box_positive_fraction=box_positive_fraction,
            bbox_reg_weights=None,
            mask_roi_pool=mask_roi_pool,
            mask_head=mask_head,
            mask_predictor=mask_predictor,
        )

    def forward(self, images, targets=None):
        if targets is not None:
            return self.model(images, targets)
        else:
            return self.model(images)
