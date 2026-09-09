"""Dataset loading, leakage controls, and SSIM de-duplication for Bio-CBAM.

FER-2013 follows the official Usage field. CK+ and JAFFE are loaded from explicit
manifests with subject identifiers. SSIM filtering is optional and is applied to
training records only; it is not presented as a substitute for subject identity.
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


FER2013_LABELS: Dict[int, str] = {
    0: "angry",
    1: "disgust",
    2: "fear",
    3: "happy",
    4: "sad",
    5: "surprise",
    6: "neutral",
}
FER2013_USAGE = {"train": "Training", "val": "PublicTest", "test": "PrivateTest"}
DEFAULT_FOUR_CLASSES = ("angry", "happy", "sad", "neutral")


@dataclass(frozen=True)
class SSIMFilterConfig:
    threshold: float = 0.95
    hash_size: int = 8
    bands: int = 4
    same_class_only: bool = True
    max_candidates_per_image: int = 256


@dataclass
class ImageRecord:
    identifier: str
    label: int
    label_name: str
    split: str
    subject_id: Optional[str] = None
    path: Optional[str] = None
    pixels: Optional[str] = None


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_transforms(image_size: int = 224) -> Dict[str, Callable]:
    normalize = transforms.Normalize(
        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
    )
    return {
        "train": transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                normalize,
            ]
        ),
        "eval": transforms.Compose(
            [transforms.Resize((image_size, image_size)), transforms.ToTensor(), normalize]
        ),
    }


def _parse_fer_pixels(pixel_string: str) -> np.ndarray:
    pixels = np.fromstring(pixel_string, sep=" ", dtype=np.uint8)
    if pixels.size != 48 * 48:
        raise ValueError(f"FER-2013 row contains {pixels.size} pixels instead of 2304")
    return pixels.reshape(48, 48)


def _ssim(gray_a: np.ndarray, gray_b: np.ndarray) -> float:
    """Local SSIM with an 11x11 Gaussian window, for deterministic auditing."""
    a = gray_a.astype(np.float64)
    b = gray_b.astype(np.float64)
    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2
    mu_a = cv2.GaussianBlur(a, (11, 11), 1.5)
    mu_b = cv2.GaussianBlur(b, (11, 11), 1.5)
    sigma_a = cv2.GaussianBlur(a * a, (11, 11), 1.5) - mu_a * mu_a
    sigma_b = cv2.GaussianBlur(b * b, (11, 11), 1.5) - mu_b * mu_b
    sigma_ab = cv2.GaussianBlur(a * b, (11, 11), 1.5) - mu_a * mu_b
    numerator = (2 * mu_a * mu_b + c1) * (2 * sigma_ab + c2)
    denominator = (mu_a * mu_a + mu_b * mu_b + c1) * (sigma_a + sigma_b + c2)
    return float(np.mean(numerator / np.maximum(denominator, 1e-12)))


def _perceptual_hash(gray: np.ndarray, hash_size: int = 8) -> int:
    resized = cv2.resize(gray, (hash_size * 4, hash_size * 4), interpolation=cv2.INTER_AREA)
    dct = cv2.dct(resized.astype(np.float32))[:hash_size, :hash_size]
    bits = dct > np.median(dct[1:])
    value = 0
    for bit in bits.flatten():
        value = (value << 1) | int(bit)
    return value


def _hash_bands(value: int, bit_count: int, bands: int) -> Iterable[Tuple[int, int]]:
    if bit_count % bands:
        raise ValueError("hash bits must be divisible by bands")
    band_width = bit_count // bands
    mask = (1 << band_width) - 1
    for band in range(bands):
        yield band, (value >> (band * band_width)) & mask


def filter_training_records_ssim(
    records: Sequence[ImageRecord],
    image_reader: Callable[[ImageRecord], np.ndarray],
    config: SSIMFilterConfig,
    audit_path: Optional[Path] = None,
) -> List[ImageRecord]:
    """Remove near-duplicates from training only and emit an auditable manifest.

    Perceptual-hash bands produce candidates; SSIM makes the final decision. The
    operation is deterministic and order preserving. No validation/test record is
    removed. Comparisons default to the same class to avoid label-dependent deletion.
    """
    if not 0 < config.threshold <= 1:
        raise ValueError("SSIM threshold must lie in (0,1]")
    if any(record.split != "train" for record in records):
        raise ValueError("SSIM filtering accepts training records only")

    bit_count = config.hash_size * config.hash_size
    buckets: Dict[Tuple[object, int, int], List[int]] = defaultdict(list)
    kept: List[ImageRecord] = []
    kept_images: List[np.ndarray] = []
    removed: List[Dict[str, object]] = []

    for record in records:
        image = image_reader(record)
        gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        fingerprint = _perceptual_hash(gray, config.hash_size)
        class_key: object = record.label if config.same_class_only else "all"
        candidates = set()
        keys = []
        for band, fragment in _hash_bands(fingerprint, bit_count, config.bands):
            key = (class_key, band, fragment)
            keys.append(key)
            candidates.update(buckets[key])
        ranked = sorted(candidates)[-config.max_candidates_per_image :]

        duplicate_of = None
        duplicate_score = None
        for kept_index in ranked:
            score = _ssim(gray, kept_images[kept_index])
            if score >= config.threshold:
                duplicate_of, duplicate_score = kept_index, score
                break

        if duplicate_of is not None:
            removed.append(
                {
                    "removed_id": record.identifier,
                    "kept_id": kept[duplicate_of].identifier,
                    "label": record.label_name,
                    "ssim": duplicate_score,
                }
            )
            continue

        new_index = len(kept)
        kept.append(record)
        kept_images.append(gray)
        for key in keys:
            buckets[key].append(new_index)

    if audit_path is not None:
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "configuration": asdict(config),
            "input_count": len(records),
            "kept_count": len(kept),
            "removed_count": len(removed),
            "removed_pairs": removed,
        }
        audit_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return kept


class RecordDataset(Dataset):
    def __init__(
        self,
        records: Sequence[ImageRecord],
        transform: Optional[Callable] = None,
        return_metadata: bool = False,
    ) -> None:
        if not records:
            raise ValueError("Dataset has no records; verify split names and paths")
        self.records = list(records)
        self.transform = transform
        self.return_metadata = return_metadata

    @staticmethod
    def read_record(record: ImageRecord) -> np.ndarray:
        if record.pixels is not None:
            gray = _parse_fer_pixels(record.pixels)
            return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        if record.path is None:
            raise ValueError(f"Record {record.identifier} has neither pixels nor path")
        image = cv2.imread(record.path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(record.path)
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        image = Image.fromarray(self.read_record(record))
        tensor = self.transform(image) if self.transform else transforms.ToTensor()(image)
        if self.return_metadata:
            metadata = {
                "identifier": record.identifier,
                "subject_id": record.subject_id or "",
                "label_name": record.label_name,
                "split": record.split,
            }
            return tensor, record.label, metadata
        return tensor, record.label


class FER2013Dataset(RecordDataset):
    """FER-2013 using the official Training/PublicTest/PrivateTest partitions."""

    def __init__(
        self,
        csv_path: str,
        split: str = "train",
        class_names: Optional[Sequence[str]] = None,
        transform: Optional[Callable] = None,
        ssim_config: Optional[SSIMFilterConfig] = None,
        ssim_audit_path: Optional[str] = None,
        return_metadata: bool = False,
    ) -> None:
        split_key = split.lower()
        if split_key not in FER2013_USAGE:
            raise ValueError("FER-2013 split must be train, val, or test")
        selected = tuple(name.lower() for name in class_names) if class_names else tuple(FER2013_LABELS.values())
        if len(set(selected)) != len(selected):
            raise ValueError("class_names contains duplicates")
        if "confusion" in selected or "confused" in selected:
            raise ValueError("FER-2013 has no official 'Confusion' class")
        unknown = set(selected) - set(FER2013_LABELS.values())
        if unknown:
            raise ValueError(f"Unknown FER-2013 classes: {sorted(unknown)}")
        label_to_output = {name: index for index, name in enumerate(selected)}

        source = Path(csv_path).expanduser().resolve()
        records: List[ImageRecord] = []
        if source.is_dir():
            usage = FER2013_USAGE[split_key]
            partition_dir = source / ("train" if usage == "Training" else "test")
            if not partition_dir.is_dir():
                raise ValueError(f"FER directory is missing {partition_dir}")
            for name in selected:
                class_dir = partition_dir / name
                if not class_dir.is_dir():
                    raise ValueError(f"FER directory is missing class folder {class_dir}")
                for image_path in sorted(path for path in class_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"}):
                    if usage != "Training" and not image_path.name.startswith(f"{usage}_"):
                        continue
                    relative = image_path.relative_to(source).as_posix()
                    records.append(
                        ImageRecord(
                            identifier=f"fer2013:{relative}",
                            label=label_to_output[name],
                            label_name=name,
                            split=split_key,
                            path=str(image_path),
                        )
                    )
        else:
            frame = pd.read_csv(source)
            required = {"emotion", "pixels", "Usage"}
            if not required.issubset(frame.columns):
                raise ValueError(f"FER CSV is missing columns: {sorted(required - set(frame.columns))}")
            frame = frame.loc[frame["Usage"] == FER2013_USAGE[split_key]].copy()
            for row_index, row in frame.iterrows():
                original_label = int(row["emotion"])
                if original_label not in FER2013_LABELS:
                    raise ValueError(f"Unknown FER label {original_label} at row {row_index}")
                name = FER2013_LABELS[original_label]
                if name not in label_to_output:
                    continue
                records.append(
                    ImageRecord(
                        identifier=f"fer2013:{row_index}",
                        label=label_to_output[name],
                        label_name=name,
                        split=split_key,
                        pixels=str(row["pixels"]),
                    )
                )
        if ssim_config is not None:
            if split_key != "train":
                raise ValueError("SSIM removal is intentionally restricted to the training split")
            records = filter_training_records_ssim(
                records,
                image_reader=RecordDataset.read_record,
                config=ssim_config,
                audit_path=Path(ssim_audit_path) if ssim_audit_path else None,
            )
        self.class_names = selected
        super().__init__(records, transform, return_metadata)


class ManifestDataset(RecordDataset):
    """Image dataset backed by an explicit subject-aware CSV manifest."""

    def __init__(
        self,
        manifest_path: str,
        split: str,
        transform: Optional[Callable] = None,
        label_map: Optional[Mapping[str, int]] = None,
        return_metadata: bool = False,
    ) -> None:
        manifest = Path(manifest_path).expanduser().resolve()
        frame = pd.read_csv(manifest, keep_default_na=False)
        required = {"path", "label", "subject_id", "split"}
        if not required.issubset(frame.columns):
            raise ValueError(f"Manifest is missing columns: {sorted(required - set(frame.columns))}")
        validate_subject_disjointness(frame)
        subset = frame.loc[frame["split"].astype(str).str.lower() == split.lower()].copy()
        if subset.empty:
            raise ValueError(f"No records for split '{split}' in {manifest}")

        labels = sorted(subset["label"].astype(str).str.lower().unique())
        mapping = dict(label_map) if label_map is not None else {name: i for i, name in enumerate(labels)}
        records: List[ImageRecord] = []
        for row_index, row in subset.iterrows():
            label_name = str(row["label"]).lower()
            if label_name not in mapping:
                raise ValueError(f"Label '{label_name}' is missing from label_map")
            image_path = Path(str(row["path"]))
            if not image_path.is_absolute():
                image_path = manifest.parent / image_path
            records.append(
                ImageRecord(
                    identifier=str(row.get("id", f"{manifest.stem}:{row_index}")),
                    label=int(mapping[label_name]),
                    label_name=label_name,
                    split=split.lower(),
                    subject_id=str(row["subject_id"]),
                    path=str(image_path.resolve()),
                )
            )
        self.label_map = mapping
        super().__init__(records, transform, return_metadata)


def validate_subject_disjointness(frame: pd.DataFrame) -> None:
    by_subject = frame.assign(
        subject_id=frame["subject_id"].astype(str),
        split=frame["split"].astype(str).str.lower(),
    ).groupby("subject_id")["split"].nunique()
    leaking = by_subject[by_subject > 1].index.tolist()
    if leaking:
        preview = ", ".join(leaking[:10])
        raise ValueError(f"Subject leakage across splits for {len(leaking)} subjects: {preview}")


def create_subject_folds(
    input_manifest: str,
    output_dir: str,
    folds: int = 5,
    seed: int = 42,
    validation_fold_offset: int = 1,
) -> List[str]:
    """Create train/val/test manifests with StratifiedGroupKFold."""
    source = Path(input_manifest).expanduser().resolve()
    frame = pd.read_csv(source, keep_default_na=False)
    required = {"path", "label", "subject_id"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Manifest is missing columns: {sorted(required - set(frame.columns))}")
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    fold_assignments = np.full(len(frame), -1, dtype=int)
    for fold, (_, test_indices) in enumerate(
        splitter.split(frame, frame["label"], groups=frame["subject_id"])
    ):
        fold_assignments[test_indices] = fold
    if np.any(fold_assignments < 0):
        raise RuntimeError("Some records did not receive a fold")

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    outputs: List[str] = []
    for test_fold in range(folds):
        val_fold = (test_fold + validation_fold_offset) % folds
        result = frame.copy()
        result["split"] = "train"
        result.loc[fold_assignments == val_fold, "split"] = "val"
        result.loc[fold_assignments == test_fold, "split"] = "test"
        validate_subject_disjointness(result)
        path = destination / f"fold_{test_fold}.csv"
        result.to_csv(path, index=False)
        outputs.append(str(path))
    return outputs


def create_dataloaders(
    dataset_name: str,
    data_path: str,
    batch_size: int = 32,
    num_workers: int = 4,
    num_classes: int = 7,
    four_class_names: Sequence[str] = DEFAULT_FOUR_CLASSES,
    image_size: int = 224,
    seed: int = 42,
    use_ssim_filtering: bool = False,
    ssim_threshold: float = 0.95,
    ssim_audit_path: Optional[str] = None,
    return_metadata: bool = False,
) -> Dict[str, DataLoader]:
    tfm = get_transforms(image_size)
    name = dataset_name.lower()
    class_names = None if num_classes == 7 else tuple(four_class_names)
    if name == "fer2013":
        train_set = FER2013Dataset(
            data_path, "train", class_names, tfm["train"],
            SSIMFilterConfig(threshold=ssim_threshold) if use_ssim_filtering else None,
            ssim_audit_path, return_metadata,
        )
        val_set = FER2013Dataset(data_path, "val", class_names, tfm["eval"], return_metadata=return_metadata)
        test_set = FER2013Dataset(data_path, "test", class_names, tfm["eval"], return_metadata=return_metadata)
    elif name in {"ckplus", "ck+", "jaffe"}:
        complete = pd.read_csv(data_path)
        labels = sorted(complete["label"].astype(str).str.lower().unique())
        label_map = {label: index for index, label in enumerate(labels)}
        train_set = ManifestDataset(data_path, "train", tfm["train"], label_map, return_metadata)
        val_set = ManifestDataset(data_path, "val", tfm["eval"], label_map, return_metadata)
        test_set = ManifestDataset(data_path, "test", tfm["eval"], label_map, return_metadata)
    else:
        raise ValueError("dataset_name must be fer2013, ckplus, or jaffe")

    generator = torch.Generator().manual_seed(seed)
    common = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "worker_init_fn": seed_worker,
        "generator": generator,
        "persistent_workers": num_workers > 0,
    }
    return {
        "train": DataLoader(train_set, shuffle=True, **common),
        "val": DataLoader(val_set, shuffle=False, **common),
        "test": DataLoader(test_set, shuffle=False, **common),
    }


def audit_cross_split_ssim(
    reference_records: Sequence[ImageRecord],
    query_records: Sequence[ImageRecord],
    image_reader: Callable[[ImageRecord], np.ndarray],
    config: SSIMFilterConfig,
    output_path: Optional[str] = None,
) -> Dict[str, object]:
    """Audit near-duplicates between disjoint splits without deleting samples.

    Candidate generation uses perceptual-hash bands; SSIM makes the final
    decision. This audit detects image similarity, not subject identity.
    """
    if not reference_records or not query_records:
        raise ValueError("Both reference_records and query_records must be non-empty")
    bit_count = config.hash_size * config.hash_size
    buckets: Dict[Tuple[object, int, int], List[int]] = defaultdict(list)
    reference_images: List[np.ndarray] = []
    for index, record in enumerate(reference_records):
        image = image_reader(record)
        gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        reference_images.append(gray)
        fingerprint = _perceptual_hash(gray, config.hash_size)
        class_key: object = record.label if config.same_class_only else "all"
        for band, fragment in _hash_bands(fingerprint, bit_count, config.bands):
            buckets[(class_key, band, fragment)].append(index)

    matches: List[Dict[str, object]] = []
    for query in query_records:
        image = image_reader(query)
        gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        fingerprint = _perceptual_hash(gray, config.hash_size)
        class_key = query.label if config.same_class_only else "all"
        candidates = set()
        for band, fragment in _hash_bands(fingerprint, bit_count, config.bands):
            candidates.update(buckets[(class_key, band, fragment)])
        for ref_index in sorted(candidates)[-config.max_candidates_per_image :]:
            score = _ssim(gray, reference_images[ref_index])
            if score >= config.threshold:
                matches.append(
                    {
                        "reference_id": reference_records[ref_index].identifier,
                        "query_id": query.identifier,
                        "label": query.label_name,
                        "ssim": score,
                    }
                )

    report: Dict[str, object] = {
        "configuration": asdict(config),
        "reference_count": len(reference_records),
        "query_count": len(query_records),
        "match_count": len(matches),
        "matches": matches,
        "caution": "Near-image similarity is not proof of subject identity.",
    }
    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
