import os
from typing import Callable

from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
import torch
import pandas as pd
from PIL import Image

from jormungandr.utils.image_processors import (
    DetrImageProcessorNoPadBBoxUpdate as DetrImageProcessor,
)
from jormungandr.utils.seed import build_torch_generator, seed_worker

model_name = "facebook/detr-resnet-50"
image_processor: DetrImageProcessor | None = None
try:
    image_processor = DetrImageProcessor.from_pretrained(
        model_name, force_download=True
    )
except Exception as e:
    print(f"Error occurred while loading image processor: {e}")
    print(
        "There is likely an issue with the Hugging Face Hub or your internet connection. If you have previously loaded the model, it should be cached locally. Attempting to load from cache..."
    )
    image_processor = DetrImageProcessor.from_pretrained(
        model_name, local_files_only=True
    )


# maps 80-class index -> 91-class index
coco80_to_coco91 = {
    0: 1,  # person
    1: 2,  # bicycle
    2: 3,  # car
    3: 4,  # motorcycle
    4: 5,  # airplane
    5: 6,  # bus
    6: 7,  # train
    7: 8,  # truck
    8: 9,  # boat
    9: 10,  # traffic light
    10: 11,  # fire hydrant
    11: 13,  # stop sign
    12: 14,  # parking meter
    13: 15,  # bench
    14: 16,  # bird
    15: 17,  # cat
    16: 18,  # dog
    17: 19,  # horse
    18: 20,  # sheep
    19: 21,  # cow
    20: 22,  # elephant
    21: 23,  # bear
    22: 24,  # zebra
    23: 25,  # giraffe
    24: 27,  # backpack
    25: 28,  # umbrella
    26: 31,  # handbag
    27: 32,  # tie
    28: 33,  # suitcase
    29: 34,  # frisbee
    30: 35,  # skis
    31: 36,  # snowboard
    32: 37,  # sports ball
    33: 38,  # kite
    34: 39,  # baseball bat
    35: 40,  # baseball glove
    36: 41,  # skateboard
    37: 42,  # surfboard
    38: 43,  # tennis racket
    39: 44,  # bottle
    40: 46,  # wine glass
    41: 47,  # cup
    42: 48,  # fork
    43: 49,  # knife
    44: 50,  # spoon
    45: 51,  # bowl
    46: 52,  # banana
    47: 53,  # apple
    48: 54,  # sandwich
    49: 55,  # orange
    50: 56,  # broccoli
    51: 57,  # carrot
    52: 58,  # hot dog
    53: 59,  # pizza
    54: 60,  # donut
    55: 61,  # cake
    56: 62,  # chair
    57: 63,  # couch
    58: 64,  # potted plant
    59: 65,  # bed
    60: 67,  # dining table
    61: 70,  # toilet
    62: 72,  # tv
    63: 73,  # laptop
    64: 74,  # mouse
    65: 75,  # remote
    66: 76,  # keyboard
    67: 77,  # cell phone
    68: 78,  # microwave
    69: 79,  # oven
    70: 80,  # toaster
    71: 81,  # sink
    72: 82,  # refrigerator
    73: 84,  # book
    74: 85,  # clock
    75: 86,  # vase
    76: 87,  # scissors
    77: 88,  # teddy bear
    78: 89,  # hair drier
    79: 90,  # toothbrush
}


def _collate_fn(batch):
    # images from dataset (CHW uint8)
    images = [_ensure_3ch(item["image"]) for item in batch]

    # build COCO-style annotations per image
    targets = []
    for item in batch:
        # dataset bbox format: [x_min, y_min, x_max, y_max] (absolute pixels)
        boxes = item["objects"]["bbox"]
        boxes[:, 2] -= boxes[:, 0]  # width = x_max - x_min
        boxes[:, 3] -= boxes[:, 1]  # height = y_max - y_min

        class_ids = item["objects"]["category"]  # (N,)
        areas = item["objects"].get("area", None)
        iscrowd = item["objects"].get("iscrowd", None)

        annotations = []
        for i in range(boxes.shape[0]):
            ann = {
                "bbox": boxes[i].tolist(),  # COCO expects python lists
                "category_id": coco80_to_coco91[int(class_ids[i].item())],
            }
            if areas is not None:
                ann["area"] = float(areas[i].item())
            if iscrowd is not None:
                ann["iscrowd"] = int(iscrowd[i].item())
            else:
                ann["iscrowd"] = 0
            annotations.append(ann)

        targets.append(
            {"image_id": int(item["image_id"].item()), "annotations": annotations}
        )

    encoded = image_processor(images=images, annotations=targets, return_tensors="pt")

    # DetrImageProcessor scales label["area"] by the resize ratio
    # (resize_annotation multiplies by ratio_w * ratio_h). COCO eval size
    # thresholds (small < 32², medium < 96²) are defined in original image
    # pixel space, so we restore each image's areas to original coordinates
    # using label["orig_size"] (original) and label["size"] (after resize).
    for label in encoded["labels"]:
        if "area" in label:
            orig_h, orig_w = label["orig_size"].tolist()
            size_h, size_w = label["size"].tolist()
            label["area"] = label["area"] * (orig_h / size_h) * (orig_w / size_w)

    return {
        "pixel_values": encoded["pixel_values"],  # (B, 3, Hmax, Wmax)
        "pixel_mask": encoded["pixel_mask"],  # (B, Hmax, Wmax)
        "labels": encoded["labels"],  # list[dict] length B
    }


def _ensure_3ch(img: torch.Tensor) -> torch.Tensor:
    # img: (C,H,W), uint8
    if img.ndim != 3:
        raise ValueError(f"Expected CHW image, got shape {tuple(img.shape)}")

    c, h, w = img.shape
    if c == 3:
        return img
    if c == 1:
        return img.repeat(3, 1, 1)
    if c == 4:
        return img[:3]  # drop alpha
    raise ValueError(f"Unsupported channel count: {c} (shape {tuple(img.shape)})")


def _collate_fn_vod(batch):
    all_images = []
    all_annotations = []
    for item in batch:
        all_images.extend(item["pixel_values"])
        all_annotations.extend(item["labels"])

    encoded = image_processor(
        images=all_images, annotations=all_annotations, return_tensors="pt"
    )

    for label in encoded["labels"]:
        if "area" in label:
            orig_h, orig_w = label["orig_size"].tolist()
            size_h, size_w = label["size"].tolist()
            label["area"] = label["area"] * (orig_h / size_h) * (orig_w / size_w)

    return {
        "pixel_values": encoded["pixel_values"],
        "pixel_mask": encoded["pixel_mask"],
        "labels": encoded["labels"],
    }


class VODDataset(Dataset):
    # sequence_dirs: list of directories, each containing frames of a video sequence
    # n_frames: number of consecutive frames to include in each clip
    # get_item should return a dict with keys "pixel_values" (list of tensors) and "labels" (list of dicts)
    # "pixel_values" should be a list of tensors of shape (Frames, Channels, Height, Width)
    # "labels" should be a list of dicts, one per frame, each containing the COCO-style annotations for that frame
    def __init__(
        self,
        sequence_dirs: list[str],
        n_frames: int = 4,
        # sampling_strategy: str = "consecutive",  # or "random"
        transform: Callable | None = None,
    ):
        self.n_frames = n_frames
        self.transform = transform
        self.sequence_dirs = sequence_dirs
        self.columns = [
            "frame_number",
            "identity_number",
            "bbox_left",
            "bbox_top",
            "bbox_width",
            "bbox_height",
            "confidence_score",
            "class",
            "visibility",
        ]
        self.clips: list[tuple[str, list[str]]] = []
        self.gt_annotations: dict[str, pd.DataFrame] = {}

        """
        Sequence to sequence predictions: for each sequence, we can create multiple clips by sliding a window of size n_frames across the frames. For example, if a sequence has 10 frames and n_frames=4, we can create the following clips:
            - Clip 1: frames 1-4
            - Clip 2: frames 2-5
            - Clip 3: frames 3-6
        They also apply normal image augmentations: random horizontal flip and random resizing
        """
        """
        sequence_dir
        ├── img1/
        │   ├── 00001.jpg
        │   ├── 00002.jpg
        │   └── ...
        ├── gt/
        │   └── gt.txt
        └── sequence.ini

        """
        for seq_dir in sequence_dirs:
            img_dir = os.path.join(seq_dir, "img1")
            frame_files = sorted(
                [
                    f
                    for f in os.listdir(img_dir)
                    if f.endswith((".jpg", ".png", ".jpeg"))
                ]
            )
            for i in range(len(frame_files) - n_frames + 1):
                clip_frames = frame_files[i : i + n_frames]
                self.clips.append((seq_dir, clip_frames))
            gt_path = os.path.join(seq_dir, "gt", "gt.txt")
            self.prepare_dataframe(seq_dir, gt_path)

    def prepare_dataframe(self, seq_dir: str, gt_path: str):
        gt = pd.read_csv(gt_path, sep=",", header=None, names=self.columns)
        filtered_gt = gt.query("confidence_score != 0").copy()
        filtered_gt.loc[:, "bbox"] = filtered_gt[
            ["bbox_left", "bbox_top", "bbox_width", "bbox_height"]
        ].values.tolist()
        filtered_gt = filtered_gt[["frame_number", "bbox", "class", "visibility"]]
        combined = (
            filtered_gt.groupby("frame_number")
            .agg({"bbox": list, "class": list, "visibility": list})
            .reset_index()
        )
        combined = combined.rename(columns={"class": "category"})
        self.gt_annotations[seq_dir] = combined

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, index):
        seq_dir: str
        clip_frames: list[str]
        seq_dir, clip_frames = self.clips[index]
        pixel_values = []
        labels = []
        for frame_file in clip_frames:
            image, annotation = self.load_image(seq_dir, frame_file)
            if self.transform is not None:
                image, annotation = self.transform(image, annotation)
            pixel_values.append(image)
            labels.append(annotation)
        # pixel_values = torch.stack(pixel_values)  # (Frames, Channels, Height, Width)
        return {"pixel_values": pixel_values, "labels": labels}

    def load_image(self, seq_dir: str, frame_file: str) -> tuple[Image.Image, dict]:
        frame_path = os.path.join(seq_dir, "img1", frame_file)
        img = Image.open(frame_path).convert("RGB")

        frame_number = int(os.path.splitext(frame_file)[0])
        gt = self.gt_annotations[seq_dir]

        rows = gt[gt["frame_number"] == frame_number]
        if rows.empty:
            return img, {"image_id": frame_number, "annotations": []}

        row = rows.iloc[0]
        annotations = [
            {
                "bbox": bbox,  # [x, y, w, h] absolute pixels
                "category_id": int(cat),
                "area": float(bbox[2] * bbox[3]),
                "iscrowd": 0,
            }
            for bbox, cat in zip(row["bbox"], row["category"])
        ]
        return img, {"image_id": frame_number, "annotations": annotations}


def _build_image_datasets(
    dataset_name: str,
    cache_dir: str,
    seed: int,
    subset_size: int | None,
):
    ds = load_dataset(dataset_name, cache_dir=cache_dir)
    train_ds = ds["train"].with_format("torch")
    val_ds = ds["val"].with_format("torch")
    if subset_size is not None:
        train_ds = train_ds.shuffle(seed=seed).select(range(subset_size))
        val_ds = val_ds.shuffle(seed=seed + 1).select(range(subset_size))
    return train_ds, val_ds


def _build_vod_datasets(
    data_dir: str,
    dataset_name: str,
    n_frames: int,
    val_split: float,
):
    data_path = os.path.join(data_dir, dataset_name.upper(), "train")
    sequence_dirs = sorted(
        os.path.join(data_path, d)
        for d in os.listdir(data_path)
        if os.path.isdir(os.path.join(data_path, d))
    )
    n_val = max(1, int(len(sequence_dirs) * val_split))
    train_dirs = sequence_dirs[:-n_val]
    val_dirs = sequence_dirs[-n_val:]
    return (
        VODDataset(train_dirs, n_frames=n_frames),
        VODDataset(val_dirs, n_frames=n_frames),
    )


def _build_loader(
    dataset,
    *,
    batch_size: int,
    shuffle: bool,
    collate_fn: Callable,
    generator,
    prefetch_factor: int | None = None,
) -> DataLoader:
    kwargs = dict(
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        worker_init_fn=seed_worker,
        generator=generator,
        num_workers=len(os.sched_getaffinity(0)),
        pin_memory=True,
    )
    if prefetch_factor is not None:
        kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(dataset, **kwargs)


_DATASET_DEFAULTS = {
    "image": {
        "dataset_name": "detection-datasets/coco",
        "collate_fn": _collate_fn,
        "batch_size": 32,
        "train_prefetch_factor": 2,
    },
    "video": {
        "dataset_name": "mot17",
        "collate_fn": _collate_fn_vod,
        "batch_size": 1,
        "train_prefetch_factor": None,
    },
}


def create_dataloaders(
    dataset_type: str = "image",
    dataset_name: str | None = None,
    data_dir: str = "../data/",
    batch_size: int | None = None,
    seed: int = 42,
    shuffle: bool = True,
    collate_fn: Callable | None = None,
    subset_size: int | None = None,  # image-only
    n_frames: int = 4,  # video-only
    val_split: float = 0.2,  # video-only
) -> tuple[DataLoader, DataLoader]:
    """Build train/val DataLoaders for either a HuggingFace image detection
    dataset or a local MOT-style video object-detection dataset.

    `dataset_type` picks the pipeline:
      - "image": `datasets.load_dataset(dataset_name)` + DETR image collator.
      - "video": `VODDataset` over `data_dir/<DATASET_NAME>/train/*` + clip collator.

    Per-type defaults (dataset name, collate_fn, batch_size, prefetch_factor)
    live in `_DATASET_DEFAULTS` and can be overridden by the matching kwarg.
    """
    if dataset_type not in _DATASET_DEFAULTS:
        raise ValueError(
            f"Unknown dataset_type {dataset_type!r}; expected 'image' or 'video'."
        )

    defaults = _DATASET_DEFAULTS[dataset_type]
    dataset_name = dataset_name or defaults["dataset_name"]
    collate_fn = collate_fn or defaults["collate_fn"]
    batch_size = batch_size if batch_size is not None else defaults["batch_size"]

    if dataset_type == "image":
        train_ds, val_ds = _build_image_datasets(
            dataset_name, data_dir, seed, subset_size
        )
    else:  # "video"
        train_ds, val_ds = _build_vod_datasets(
            data_dir, dataset_name, n_frames, val_split
        )

    train_gen = build_torch_generator(seed)
    val_gen = build_torch_generator(seed + 1)

    train_loader = _build_loader(
        train_ds,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        generator=train_gen,
        prefetch_factor=defaults["train_prefetch_factor"],
    )
    val_loader = _build_loader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        generator=val_gen,
    )
    return train_loader, val_loader
