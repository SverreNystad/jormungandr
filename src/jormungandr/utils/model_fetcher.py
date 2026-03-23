from functools import lru_cache
from transformers import DetrForObjectDetection, DetrConfig


@lru_cache(maxsize=None)
def fetch_detr_model(
    model_name: str = "facebook/detr-resnet-50",
    is_pre_trained: bool = True,
    num_encoder_layers: int | None = None,
) -> DetrForObjectDetection:
    config = DetrConfig.from_pretrained(model_name)

    if num_encoder_layers is not None:
        config.encoder_layers = num_encoder_layers

    if not is_pre_trained:
        return DetrForObjectDetection(config)
    return DetrForObjectDetection.from_pretrained(model_name)
