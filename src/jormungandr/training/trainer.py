"""
The trainer script is responsible for training the Jormungandr model. It includes functions for loading data, defining the training loop, and evaluating the model's performance. The trainer will utilize the output head defined in `output_head.py` to compute the loss and update the model's parameters during training. It will also handle saving and loading model checkpoints, as well as logging training metrics for analysis.
It shall be able to train the model on a given dataset, and evaluate its performance on a validation set. The trainer will also include functionality for hyperparameter tuning and early stopping based on validation performance. Additionally, it will support distributed training across multiple GPUs if available.
It shall be able to customize the training process with different optimizers, learning rate schedulers, and loss functions. The trainer will also include functionality for visualizing training progress and results, such as plotting loss curves and displaying sample predictions on the validation set. Overall, the trainer will be a crucial component in the development and optimization of the Jormungandr model for object detection tasks.
It shall be able to train both the Fafnir and Jormungandr models

It shall use:
* Torch
* Torchvision for data loading and augmentation
* Freeze backbone
* Learning rate schedular
* Different LR for encoder and decoder
* MultiGPU support Accelerate
* Wandb for logging and visualization
    - Log epoch-wise training and validation loss
    - Log validation metrics
        - mAP, precision, recall, etc.
        - CIoU, GIoU, etc.
    - Log validation images with predicted bounding boxes and labels
    - Log model checkpoints and hyperparameters
        - Learning rate, gradient norms, etc.
    - Watch model gradients and parameters

*
"""

from tqdm import tqdm, trange
from typing import Callable
import numpy as np
import wandb
import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_
from datetime import datetime
from transformers.image_transforms import center_to_corners_format

from jormungandr.config.configuration import Config, load_config
from jormungandr.dataset import create_dataloaders
from jormungandr.fafnir import Fafnir
from jormungandr.training.criterion import build_criterion
from jormungandr.training.coco_eval import CocoEvaluator

_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225])

COCO_ID_TO_NAME = {
    1: "person",
    2: "bicycle",
    3: "car",
    4: "motorcycle",
    5: "airplane",
    6: "bus",
    7: "train",
    8: "truck",
    9: "boat",
    10: "traffic light",
    11: "fire hydrant",
    13: "stop sign",
    14: "parking meter",
    15: "bench",
    16: "bird",
    17: "cat",
    18: "dog",
    19: "horse",
    20: "sheep",
    21: "cow",
    22: "elephant",
    23: "bear",
    24: "zebra",
    25: "giraffe",
    27: "backpack",
    28: "umbrella",
    31: "handbag",
    32: "tie",
    33: "suitcase",
    34: "frisbee",
    35: "skis",
    36: "snowboard",
    37: "sports ball",
    38: "kite",
    39: "baseball bat",
    40: "baseball glove",
    41: "skateboard",
    42: "surfboard",
    43: "tennis racket",
    44: "bottle",
    46: "wine glass",
    47: "cup",
    48: "fork",
    49: "knife",
    50: "spoon",
    51: "bowl",
    52: "banana",
    53: "apple",
    54: "sandwich",
    55: "orange",
    56: "broccoli",
    57: "carrot",
    58: "hot dog",
    59: "pizza",
    60: "donut",
    61: "cake",
    62: "chair",
    63: "couch",
    64: "potted plant",
    65: "bed",
    67: "dining table",
    70: "toilet",
    72: "tv",
    73: "laptop",
    74: "mouse",
    75: "remote",
    76: "keyboard",
    77: "cell phone",
    78: "microwave",
    79: "oven",
    80: "toaster",
    81: "sink",
    82: "refrigerator",
    84: "book",
    85: "clock",
    86: "vase",
    87: "scissors",
    88: "teddy bear",
    89: "hair drier",
    90: "toothbrush",
}

CONFIG = load_config("config.yaml")
MODELS_PATH = "models/"
# Initializing in a separate cell so we can easily add more epochs to the same run
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")


def _log_validation_images(
    pixel_values: torch.Tensor,  # [B, 3, H, W] normalized
    pixel_mask: torch.Tensor,  # [B, H, W]
    labels: list[dict],
    class_logits: torch.Tensor,  # [B, Q, num_classes+1]
    pred_boxes: torch.Tensor,  # [B, Q, 4] normalized cxcywh
    num_images: int = 8,
    score_threshold: float = 0.5,
) -> list[wandb.Image]:
    probs = class_logits.softmax(-1).cpu()
    pred_boxes = pred_boxes.cpu()
    pixel_values = pixel_values.cpu()
    pixel_mask = pixel_mask.cpu()

    # Best foreground class per query (exclude no-object = last index)
    scores, pred_classes = probs[..., :-1].max(-1)  # [B, Q]

    wandb_images = []
    for b in range(min(num_images, len(labels))):
        # Crop padding: find actual image dimensions from pixel_mask
        actual_h = int(pixel_mask[b].any(dim=-1).sum().item())
        actual_w = int(pixel_mask[b].any(dim=-2).sum().item())

        # Denormalize: (3, H, W) float -> (H, W, 3) uint8, crop padding
        img = (
            pixel_values[b] * _IMAGENET_STD[:, None, None]
            + _IMAGENET_MEAN[:, None, None]
        )
        img = (img.clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        img = img[:actual_h, :actual_w]

        # GT boxes: normalized cxcywh -> pixel xyxy
        gt_boxes_norm = labels[b]["boxes"].cpu()
        gt_classes = labels[b]["class_labels"].cpu()
        gt_xyxy = center_to_corners_format(gt_boxes_norm)
        gt_xyxy[:, [0, 2]] *= actual_w
        gt_xyxy[:, [1, 3]] *= actual_h

        gt_box_data = [
            {
                "position": {
                    "minX": float(x1),
                    "maxX": float(x2),
                    "minY": float(y1),
                    "maxY": float(y2),
                },
                "class_id": int(gt_classes[i].item()),
                "box_caption": COCO_ID_TO_NAME.get(int(gt_classes[i].item()), str(int(gt_classes[i].item()))),
                "domain": "pixel",
            }
            for i, (x1, y1, x2, y2) in enumerate(gt_xyxy.tolist())
        ]

        # Predicted boxes: filter by score threshold, normalized cxcywh -> pixel xyxy
        pred_xyxy = center_to_corners_format(pred_boxes[b])
        pred_xyxy[:, [0, 2]] *= actual_w
        pred_xyxy[:, [1, 3]] *= actual_h

        pred_box_data = []
        for q, (x1, y1, x2, y2) in enumerate(pred_xyxy.tolist()):
            s = float(scores[b, q].item())
            if s < score_threshold:
                continue
            cls_id = int(pred_classes[b, q].item())
            cls_name = COCO_ID_TO_NAME.get(cls_id, str(cls_id))
            pred_box_data.append(
                {
                    "position": {"minX": x1, "maxX": x2, "minY": y1, "maxY": y2},
                    "class_id": cls_id,
                    "box_caption": f"{cls_name}: {s:.2f}",
                    "scores": {"confidence": round(s, 3)},
                    "domain": "pixel",
                }
            )

        wandb_images.append(
            wandb.Image(
                img,
                boxes={
                    "ground_truth": {
                        "box_data": gt_box_data,
                        "class_labels": COCO_ID_TO_NAME,
                    },
                    "predictions": {
                        "box_data": pred_box_data,
                        "class_labels": COCO_ID_TO_NAME,
                    },
                },
            )
        )

    return wandb_images


def train(
    config: Config,
):
    device = "cuda"
    model = Fafnir(config=config.fafnir).to(device)
    wandb.watch(model, log="all", log_freq=100)
    training_loader, validation_loader = create_dataloaders(
        batch_size=config.trainer.batch_size,
        seed=config.trainer.seed,
        subset_size=100,
    )

    criterion = build_criterion(config.trainer.loss.name)
    optimizer = AdamW(
        [
            {
                "params": model.encoder.parameters(),
                "lr": config.trainer.encoder_learning_rate,
            },
            {
                "params": model.decoder.parameters(),
                "lr": config.trainer.decoder_learning_rate,
            },
            {
                "params": model.output_head.parameters(),
                "lr": config.trainer.output_head_learning_rate,
            },
        ]
    )

    best_val_loss = float("inf")
    for epoch in trange(config.trainer.epochs, desc="Epochs", unit="epoch"):
        average_training_loss = train_one_epoch(
            model,
            training_loader,
            optimizer,
            criterion,
            device=device,
            config=config,
        )
        average_validation_loss, average_validation_time = run_validation(
            model,
            validation_loader,
            criterion,
            device=device,
            config=config,
        )

        wandb.log(
            {
                "avg_train_loss": average_training_loss,
                "avg_val_loss": average_validation_loss,
                "epoch": epoch,
                "avg_val_time": average_validation_time,
            }
        )

        # Track best performance, and save the model's state
        if average_validation_loss < best_val_loss:
            best_val_loss = average_validation_loss
            artifact_name = f"model_{timestamp}_{best_val_loss:.3f}_{epoch}"
            model_path = f"{MODELS_PATH}{artifact_name}"
            torch.save(model.state_dict(), model_path)

            model_artifact = wandb.Artifact(
                artifact_name,
                type="model",
            )
            model_artifact.add_file(model_path)
            wandb.log_artifact(model_artifact)

    return model


def train_one_epoch(
    model: Fafnir,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module | Callable,
    device: torch.device | str,
    config: Config,
) -> float:
    model.train(True)

    running_loss = 0.0

    for i, data in tqdm(
        enumerate(dataloader), desc="Batches", unit="batch", leave=False
    ):
        pixel_values, pixel_mask, labels = (
            data["pixel_values"],
            data["pixel_mask"],
            data["labels"],
        )
        pixel_values = pixel_values.to(device, non_blocking=True)
        pixel_mask = pixel_mask.to(device, non_blocking=True)
        labels = [
            {k: v.to(device, non_blocking=True) for k, v in label.items()}
            for label in labels
        ]

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        class_labels, bbox_coordinates = model.forward(pixel_values)
        loss, loss_dict, auxiliary_outputs = criterion(
            logits=class_labels,
            labels=labels,
            device=device,
            pred_boxes=bbox_coordinates,
            config=config.trainer.loss,
        )

        # Backward pass and optimize
        loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        batch_loss = loss.item()
        running_loss += batch_loss
        wandb.log(
            {
                "train/batch_loss": batch_loss,
                **{f"train/loss/{k}": v for k, v in loss_dict.items()},
                # **{f"batch/aux/{k}": v for k, v in auxiliary_outputs.items()},
            }
        )
    average_loss = running_loss / (i + 1)
    return average_loss


# Disable gradient computation and reduce memory consumption.
@torch.no_grad()
def run_validation(
    model: Fafnir,
    validation_loader: DataLoader,
    criterion: nn.Module | Callable,
    device: torch.device | str,
    config: Config = CONFIG,
) -> tuple[float, float]:
    running_val_loss = 0.0
    # Set the model to evaluation mode, disabling dropout and using population
    # statistics for batch normalization.
    model.eval()

    timings = []
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    evaluator = CocoEvaluator()
    running_loss_dict: dict[str, float] = {}
    viz_batch: dict | None = None

    for i, batch in enumerate(validation_loader):
        pixel_values, pixel_mask, labels = (
            batch["pixel_values"],
            batch["pixel_mask"],
            batch["labels"],
        )
        pixel_values = pixel_values.to(device, non_blocking=True)
        pixel_mask = pixel_mask.to(device, non_blocking=True)
        labels = [
            {k: v.to(device, non_blocking=True) for k, v in label.items()}
            for label in labels
        ]

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        starter.record()
        class_labels, bbox_coordinates = model(pixel_values)
        ender.record()

        torch.cuda.synchronize()
        timings.append(starter.elapsed_time(ender))

        val_loss, loss_dict, auxiliary_outputs = criterion(
            logits=class_labels,
            labels=labels,
            device=device,
            pred_boxes=bbox_coordinates,
            config=config.trainer.loss,
        )

        # Aggregate validation loss and metrics
        batch_loss = val_loss.item()
        running_val_loss += batch_loss

        for k, v in loss_dict.items():
            running_loss_dict[k] = running_loss_dict.get(k, 0.0) + v

        evaluator.update(class_labels, bbox_coordinates, labels)

        # Stash the first batch for image logging
        if viz_batch is None:
            n = config.trainer.num_log_images
            viz_batch = {
                "pixel_values": pixel_values[:n].cpu(),
                "pixel_mask": pixel_mask[:n].cpu(),
                "labels": [{k: v.cpu() for k, v in lbl.items()} for lbl in labels[:n]],
                "class_logits": class_labels[:n].cpu(),
                "pred_boxes": bbox_coordinates[:n].cpu(),
            }

    coco_metrics = evaluator.evaluate()
    average_loss_dict = {k: v / (i + 1) for k, v in running_loss_dict.items()}

    wandb_images = _log_validation_images(
        **viz_batch,
        num_images=config.trainer.num_log_images,
        score_threshold=config.trainer.viz_score_threshold,
    )
    wandb.log(
        {
            "val/images": wandb_images,
            **{f"val/loss/{k}": v for k, v in average_loss_dict.items()},
            **{f"val/metrics/{k}": v for k, v in coco_metrics.items()},
        }
    )

    average_time = sum(timings) / len(timings)
    average_val_loss = running_val_loss / (i + 1)
    return average_val_loss, average_time
