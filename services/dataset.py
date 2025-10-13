from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd

from utils.logger import get_logger

LOGGER = get_logger(__name__)


class DatasetManager:
    def __init__(self, dataset_path: Path | str):
        self.dataset_path = Path(dataset_path)
        self._ensure_exists()
    
    def _ensure_exists(self) -> None:
        if not self.dataset_path.exists():
            LOGGER.warning(f"Dataset not found, creating: {self.dataset_path}")
            self.dataset_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(columns=["message", "label"]).to_csv(self.dataset_path, index=False)
    
    def add_sample(self, message: str, label: int) -> bool:
        if label not in (0, 1):
            raise ValueError("Label must be 0 or 1")
        
        try:
            df = pd.read_csv(self.dataset_path)
            duplicate_mask = (df["message"] == message) & (df["label"] == label)
            
            if duplicate_mask.any():
                LOGGER.debug("Sample already exists")
                return False
            
            df.loc[len(df)] = [message, label]
            df.to_csv(self.dataset_path, index=False)
            LOGGER.info(f"Added sample: {message[:50]}... | label={label}")
            return True
        except Exception as e:
            LOGGER.error(f"Failed to add sample: {e}")
            return False
    
    def get_row_count(self) -> int:
        try:
            with open(self.dataset_path, newline="", encoding="utf-8") as f:
                return sum(1 for _ in csv.reader(f)) - 1
        except FileNotFoundError:
            return 0
    
    def get_size_kb(self) -> int:
        if self.dataset_path.exists():
            return self.dataset_path.stat().st_size // 1024
        return 0
