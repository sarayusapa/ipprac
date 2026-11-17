"""Backbones for the 3D RetinaNet nodule detector (MONAI detection).

  resnet50   MONAI LUNA16 tutorial reference: 3D ResNet-50 + FPN, trained from scratch
  ctfm       CT-FM SegResEncoder (77.8M), self-supervised on 148k CTs  (project-lighter/ct_fm_feature_extractor)
  ctfm_wbs   same encoder after supervised whole-body segmentation, 117 structures
                                                                     (project-lighter/whole_body_segmentation)
  ctfm_rand  same architecture, random init (control: does pretraining help?)

The detection pipeline works in RAS layout (X, Y, Z). CT-FM was pretrained on SPL
(Z first, posterior, left), so the CT-FM wrapper permutes/flips the input to SPL and
maps every feature map back to RAS, so anchors and boxes stay in the pipeline's frame.
"""
from __future__ import annotations

from collections import OrderedDict

import torch
from torch import nn

CTFM_CHANNELS = [32, 64, 128, 256, 512]  # SegResEncoder levels, strides 1, 2, 4, 8, 16
FPN_CHANNELS = 256


def _ctfm_encoder(kind: str) -> nn.Module:
    from lighter_zoo import SegResEncoder, SegResNet

    if kind == "ctfm":
        return SegResEncoder.from_pretrained("project-lighter/ct_fm_feature_extractor")
    if kind == "ctfm_wbs":
        return SegResNet.from_pretrained("project-lighter/whole_body_segmentation").encoder
    if kind == "ctfm_rand":
        from monai.networks.nets.segresnet_ds import SegResEncoder as MonaiSegResEncoder

        return MonaiSegResEncoder(spatial_dims=3, in_channels=1, init_filters=32, blocks_down=(1, 2, 2, 4, 4))
    raise ValueError(kind)


class CTFMFeatureExtractor(nn.Module):
    """SegResEncoder levels (default strides 4 and 8) + FPN + extra max-pool level (stride 16),
    the same 3-level layout as the MONAI LUNA16 tutorial."""

    def __init__(self, kind: str, levels=(2, 3), to_spl: bool = True):
        super().__init__()
        from monai.networks.blocks.feature_pyramid_network import FeaturePyramidNetwork, LastLevelMaxPool

        self.body = _ctfm_encoder(kind)
        self.levels = list(levels)
        self.to_spl = to_spl
        self.fpn = FeaturePyramidNetwork(
            spatial_dims=3,
            in_channels_list=[CTFM_CHANNELS[i] for i in self.levels],
            out_channels=FPN_CHANNELS,
            extra_blocks=LastLevelMaxPool(3),
        )
        self.out_channels = FPN_CHANNELS

    def forward(self, x: torch.Tensor):
        if self.to_spl:  # RAS (B,C,X,Y,Z) -> SPL (B,C,Z,Y,X) with Y,X flipped
            x = x.permute(0, 1, 4, 3, 2).flip(3, 4)
        feats = self.body(x)
        sel = OrderedDict()
        for i in self.levels:
            f = feats[i]
            if self.to_spl:
                f = f.flip(3, 4).permute(0, 1, 4, 3, 2)
            sel[str(i)] = f
        return self.fpn(sel)


def build_detector(backbone: str, spatial_dims: int = 3):
    """Returns (detector, size_divisible). Settings follow the MONAI LUNA16 16 GB config."""
    from monai.apps.detection.networks.retinanet_detector import RetinaNetDetector
    from monai.apps.detection.networks.retinanet_network import RetinaNet, resnet_fpn_feature_extractor
    from monai.apps.detection.utils.anchor_utils import AnchorGeneratorWithAnchorShape
    from monai.networks.nets import resnet

    returned_layers = [1, 2]
    anchor_generator = AnchorGeneratorWithAnchorShape(
        feature_map_scales=[2**l for l in range(len(returned_layers) + 1)],
        base_anchor_shapes=[[6, 8, 4], [8, 6, 5], [10, 10, 6]],
    )
    if backbone == "resnet50":
        conv1_t_stride = [2, 2, 1]
        body = resnet.ResNet(
            block=resnet.ResNetBottleneck, layers=[3, 4, 6, 3], block_inplanes=resnet.get_inplanes(),
            n_input_channels=1, conv1_t_stride=conv1_t_stride,
            conv1_t_size=[max(7, 2 * s + 1) for s in conv1_t_stride],
        )
        fe = resnet_fpn_feature_extractor(backbone=body, spatial_dims=spatial_dims, pretrained_backbone=False,
                                          trainable_backbone_layers=None, returned_layers=returned_layers)
        size_divisible = [s * 2 * 2 ** max(returned_layers) for s in fe.body.conv1.stride]
    else:
        fe = CTFMFeatureExtractor(backbone)
        size_divisible = [16, 16, 16]  # encoder reaches stride 16 on every axis
    net = RetinaNet(spatial_dims=spatial_dims, num_classes=1,
                    num_anchors=anchor_generator.num_anchors_per_location()[0],
                    feature_extractor=fe, size_divisible=size_divisible)
    det = RetinaNetDetector(network=net, anchor_generator=anchor_generator)
    det.set_atss_matcher(num_candidates=4, center_in_gt=False)
    det.set_hard_negative_sampler(batch_size_per_image=64, positive_fraction=0.3, pool_size=20, min_neg=16)
    det.set_target_keys(box_key="box", label_key="label")
    det.set_box_selector_parameters(score_thresh=0.02, topk_candidates_per_level=1000, nms_thresh=0.22,
                                    detections_per_img=100)
    return det, size_divisible
