"""
CNN backbone wrapper for DETR-style feature extraction.

Extracts a single-scale feature map from an image and projects it to the
model dimension. Swapping to a different backbone (ResNet-50, ResNet-101,
Swin, etc.) only requires changing the DETR model name — the rest of the
architecture is unaffected.

Classes:
    Backbone -- wraps a HuggingFace DETR backbone and its input projection layer.
"""

import torch
from torch import nn, Tensor

from jormungandr.utils.model_fetcher import fetch_detr_model


class Backbone(nn.Module):
    def __init__(
        self,
        model_name: str = "facebook/detr-resnet-50",
        freeze_backbone: bool = True,
    ):
        super(Backbone, self).__init__()
        self.backbone = fetch_detr_model(model_name).model.backbone
        self.input_projection = fetch_detr_model(model_name).model.input_projection
        self.freeze_backbone = freeze_backbone
        if self.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            for param in self.input_projection.parameters():
                param.requires_grad = False

    def forward(
        self,
        pixel_values: Tensor,
        pixel_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """
        Args:
            pixel_values: Tensor of shape (batch_size, channels, height, width)
        Returns:
            features (batch_size, h_0, w_0, model_dimension):
                Features from the backbone model.
            mask (batch_size, h_0, w_0):
                Mask indicating valid features.
        """
        batch_size, channels, height, width = pixel_values.shape
        device = pixel_values.device

        if pixel_mask is None:
            pixel_mask = torch.ones((batch_size, height, width), device=device)

        vision_features = self.backbone(pixel_values, pixel_mask)
        feature_maps, mask = vision_features[-1]

        return feature_maps, mask

    def project_feature_maps(self, feature_maps: Tensor) -> Tensor:
        projected_feature_maps = self.input_projection(feature_maps)
        return projected_feature_maps
