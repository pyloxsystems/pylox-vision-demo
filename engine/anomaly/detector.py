"""Anomaly Detector — learns "normal" per camera, scores new frames.

Uses PatchCore (fast inference, positive-only training):
- Training: Collect 200-500 "normal" frames per camera → train model on Spark
- Inference: Score each new snapshot → anomaly score 0.0-1.0
- Models stored in /data/models/{camera}/

The Spark is the "factory" — training happens once.
The Jetson loads pre-trained models for inference only.
"""

import os
import time
import logging
import numpy as np
from pathlib import Path
from typing import Optional

import torch
from PIL import Image

logger = logging.getLogger("pylox-v2.anomaly")

MODELS_DIR = Path(__file__).parent.parent.parent / "models"
FRAMES_DIR = Path(__file__).parent.parent.parent / "data" / "training_frames"


class AnomalyDetector:
    """Per-camera anomaly detection using PatchCore."""

    def __init__(self):
        self.models = {}  # camera -> loaded model
        self.thresholds = {}  # camera -> anomaly threshold
        self._stats = {
            "frames_scored": 0,
            "anomalies_detected": 0,
            "models_loaded": 0,
        }

    def load_models(self):
        """Load all trained models from disk."""
        MODELS_DIR.mkdir(parents=True, exist_ok=True)

        for cam_dir in MODELS_DIR.iterdir():
            if cam_dir.is_dir():
                camera = cam_dir.name
                model_path = cam_dir / "model.pt"
                if model_path.exists():
                    try:
                        self._load_model(camera, model_path)
                        logger.info(f"Loaded anomaly model for {camera}")
                    except Exception as e:
                        logger.error(f"Failed to load model for {camera}: {e}")

        self._stats["models_loaded"] = len(self.models)
        logger.info(f"Loaded {len(self.models)} anomaly models")

    def _load_model(self, camera: str, model_path: Path):
        """Load a trained PatchCore model. Keep memory bank on CPU to save GPU RAM."""
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        model_data = checkpoint["model"]
        # Keep memory bank on CPU (large tensor, distance computed on CPU)
        model_data["memory_bank"] = model_data["memory_bank"].cpu()
        # Feature extractor stays on CPU, moved to GPU per-inference
        self.models[camera] = model_data
        self.thresholds[camera] = checkpoint.get("threshold", 0.5)

    def score_frame(self, camera: str, frame: np.ndarray) -> Optional[float]:
        """Score a frame for anomalousness. Returns 0.0-1.0 or None if no model."""
        if camera not in self.models:
            return None

        try:
            model_data = self.models[camera]
            img = self._preprocess(frame)

            with torch.no_grad():
                # Move feature extractor to GPU for inference
                feature_extractor = model_data["feature_extractor"]
                feature_extractor.eval()
                feature_extractor.cuda()
                feature_maps = feature_extractor(img)

                # Concatenate and pool feature maps
                features_list = []
                for layer_name in sorted(feature_maps.keys()):
                    feat = feature_maps[layer_name]
                    feat = torch.nn.functional.adaptive_avg_pool2d(feat, (8, 8))
                    feat = feat.reshape(feat.shape[0], feat.shape[1], -1)
                    features_list.append(feat)

                features = torch.cat(features_list, dim=1)
                features = features.permute(0, 2, 1).reshape(-1, features.shape[1])

                # Move to CPU for distance computation (saves GPU memory)
                features = features.cpu()
                feature_extractor.cpu()
                torch.cuda.empty_cache()

                # Distance to memory bank (both on CPU)
                memory_bank = model_data["memory_bank"]
                distances = torch.cdist(features, memory_bank)
                min_distances, _ = distances.min(dim=1)

                # Score = max nearest-neighbor distance across all patches
                raw_score = min_distances.max().item()

                # Normalize: baseline normal frames score 43-65 raw_max
                # Anything above 80 is genuinely anomalous
                score = max(0.0, min(1.0, (raw_score - 50.0) / 50.0))

            self._stats["frames_scored"] += 1
            if score > self.thresholds.get(camera, 0.5):
                self._stats["anomalies_detected"] += 1

            return float(round(score, 4))

        except Exception as e:
            logger.error(f"Anomaly scoring failed for {camera}: {e}")
            return None

    def score_image_file(self, camera: str, image_path: str) -> Optional[float]:
        """Score an image file for anomalousness."""
        if camera not in self.models:
            return None
        try:
            img = np.array(Image.open(image_path).convert("RGB"))
            return self.score_frame(camera, img)
        except Exception as e:
            logger.error(f"Failed to score image {image_path}: {e}")
            return None

    def _preprocess(self, frame: np.ndarray) -> torch.Tensor:
        """Preprocess frame for PatchCore inference."""
        img = Image.fromarray(frame).resize((256, 256))
        tensor = torch.from_numpy(np.array(img)).float() / 255.0
        tensor = tensor.permute(2, 0, 1).unsqueeze(0)  # [1, 3, 256, 256]
        # Normalize with ImageNet stats
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        tensor = (tensor - mean) / std
        return tensor.cuda()

    def _compute_anomaly_score(self, features: torch.Tensor,
                                memory_bank: torch.Tensor) -> float:
        """Compute anomaly score via k-NN distance to memory bank."""
        # Flatten features
        feat_flat = features.reshape(1, -1)
        # Compute cosine distance to all memory bank entries
        distances = torch.cdist(feat_flat, memory_bank)
        # Score = distance to nearest neighbor (normalized)
        min_dist = distances.min().item()
        # Normalize to 0-1 range (calibrated during training)
        return min(1.0, min_dist / 10.0)

    def get_stats(self) -> dict:
        return {
            **self._stats,
            "cameras_with_models": list(self.models.keys()),
        }

    # --- Training Pipeline (run on Spark only) ---

    @staticmethod
    def collect_training_frames(camera: str, frigate_api: str,
                                 count: int = 300, interval_sec: int = 10):
        """Collect normal frames from Frigate for training.

        Run this during known-safe periods (e.g., business hours with normal activity).
        """
        import requests

        save_dir = FRAMES_DIR / camera
        save_dir.mkdir(parents=True, exist_ok=True)

        collected = 0
        logger.info(f"Collecting {count} training frames for {camera}...")

        for i in range(count):
            try:
                resp = requests.get(
                    f"{frigate_api}/api/{camera}/latest.jpg",
                    params={"h": 480},
                    timeout=5,
                )
                if resp.status_code == 200:
                    frame_path = save_dir / f"frame_{i:04d}.jpg"
                    frame_path.write_bytes(resp.content)
                    collected += 1
                    if collected % 50 == 0:
                        logger.info(f"  Collected {collected}/{count} frames for {camera}")
                time.sleep(interval_sec)
            except Exception as e:
                logger.warning(f"Frame collection error: {e}")

        logger.info(f"Collected {collected} frames for {camera}")
        return collected

    @staticmethod
    def train_model(camera: str):
        """Train PatchCore model on collected frames.

        This is a one-time operation — run on Spark, deploy model to Jetson.
        Takes ~5-10 minutes per camera with 300 frames.
        """
        from anomalib.models import Patchcore
        from anomalib.data import Folder
        from anomalib.engine import Engine as AnomalibEngine

        frames_dir = FRAMES_DIR / camera
        if not frames_dir.exists() or len(list(frames_dir.glob("*.jpg"))) < 50:
            raise ValueError(f"Need at least 50 training frames for {camera}, "
                           f"got {len(list(frames_dir.glob('*.jpg')))}")

        model_dir = MODELS_DIR / camera
        model_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Training PatchCore model for {camera}...")

        # Configure PatchCore
        model = Patchcore(
            backbone="wide_resnet50_2",
            layers=("layer2", "layer3"),
            coreset_sampling_ratio=0.1,
        )

        # Set up data module with folder of normal images
        datamodule = Folder(
            name=camera,
            root=str(FRAMES_DIR),
            normal_dir=camera,
            train_batch_size=32,
            eval_batch_size=32,
            num_workers=4,
        )

        # Train
        engine = AnomalibEngine(
            accelerator="gpu",
            devices=1,
            default_root_dir=str(model_dir / "logs"),
        )
        engine.fit(model=model, datamodule=datamodule)

        # Extract and save what we need for inference
        # In Anomalib v2, the inner model has the feature extractor and memory bank
        inner = model.model
        torch.save({
            "model": {
                "feature_extractor": inner.feature_extractor,
                "memory_bank": inner.memory_bank,
            },
            "threshold": 0.5,
            "camera": camera,
            "trained_at": time.time(),
            "frame_count": len(list(frames_dir.glob("*.jpg"))),
        }, model_dir / "model.pt")

        logger.info(f"Model saved for {camera} at {model_dir / 'model.pt'}")
        return str(model_dir / "model.pt")
