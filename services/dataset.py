from __future__ import annotations

import csv
from pathlib import Path

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
            with open(self.dataset_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["message", "label"])
    
    def add_sample(self, message: str, label: int) -> bool:
        if label not in (0, 1):
            raise ValueError("Label must be 0 or 1")
        
        try:
            with open(self.dataset_path, newline="", encoding="utf-8") as f:
                reader = csv.reader(f)
                next(reader, None)
                for row in reader:
                    if len(row) != 2:
                        continue
                    existing_message, existing_label = row
                    try:
                        existing_label_int = int(existing_label)
                    except ValueError:
                        continue
                    if existing_message == message and existing_label_int == label:
                        LOGGER.debug("Sample already exists")
                        return False

            with open(self.dataset_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([message, label])

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
