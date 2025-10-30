from __future__ import annotations

from pathlib import Path

import pandas as pd
from joblib import dump, load
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from core.types import FilterResult
from filters.base import BaseFilter
from utils.logger import get_logger

LOGGER = get_logger(__name__)


class TfidfFilter(BaseFilter):
    def __init__(
        self,
        model_path: Path | str | None = None,
        dataset_path: Path | str | None = None,
    ):
        super().__init__("tfidf")
        
        root_dir = Path(__file__).resolve().parents[1]
        self.model_path = Path(model_path) if model_path else root_dir / "models" / "tfidf_model.pkl"
        self.dataset_path = Path(dataset_path) if dataset_path else root_dir / "data" / "messages.csv"
        
        self.model: Pipeline | None = None
        self._new_samples = 0
        self._load_or_train()
    
    def _load_or_train(self) -> None:
        if self.model_path.exists():
            try:
                self.model = load(self.model_path)
                LOGGER.info(f"Loaded TF-IDF model from {self.model_path}")
            except Exception as e:
                LOGGER.error(f"Failed to load model: {e}, training from scratch")
                self.train()
        else:
            LOGGER.warning(f"Model not found at {self.model_path}, training from scratch")
            self.train()
    
    def train(self) -> None:
        if not self.dataset_path.exists():
            LOGGER.error(f"Dataset not found: {self.dataset_path}")
            return
        
        try:
            df = pd.read_csv(self.dataset_path)
            if len(df) == 0:
                LOGGER.error("Dataset is empty")
                return
            
            X, y = df["message"], df["label"]
            
            pipeline = Pipeline([
                ("tfidf", TfidfVectorizer(stop_words=None)),
                ("clf", MultinomialNB()),
            ])
            
            LOGGER.info(f"Training TF-IDF model on {len(df)} samples...")
            pipeline.fit(X, y)
            self.model = pipeline
            
            self.model_path.parent.mkdir(parents=True, exist_ok=True)
            dump(pipeline, self.model_path)
            LOGGER.info(f"Model saved to {self.model_path}")
            
            self._new_samples = 0
        except Exception as e:
            LOGGER.error(f"Failed to train model: {e}")
    
    async def analyze(self, text: str) -> FilterResult:
        if not self.model:
            LOGGER.warning("Model not loaded, returning neutral score")
            return FilterResult(
                filter_name=self.name,
                score=0.5,
                confidence=0.0,
                details={"error": "Model not loaded"}
            )
        
        try:
            prediction = int(self.model.predict([text])[0])
            proba = self.model.predict_proba([text])[0]
            spam_proba = float(proba[1]) if len(proba) > 1 else float(prediction)
            
            return FilterResult(
                filter_name=self.name,
                score=spam_proba,
                confidence=max(proba),
                details={
                    "prediction": prediction,
                    "probabilities": proba.tolist()
                }
            )
        except Exception as e:
            LOGGER.error(f"Prediction failed: {e}")
            return FilterResult(
                filter_name=self.name,
                score=0.5,
                confidence=0.0,
                details={"error": str(e)}
            )
    
    def is_ready(self) -> bool:
        return self.model is not None
    
    def update_dataset(self, message: str, label: int) -> bool:
        if label not in (0, 1):
            raise ValueError("Label must be 0 or 1")
        
        if not self.dataset_path.exists():
            LOGGER.error(f"Dataset not found: {self.dataset_path}")
            return False
        
        try:
            df = pd.read_csv(self.dataset_path)
            duplicate_mask = (df["message"] == message) & (df["label"] == label)
            if duplicate_mask.any():
                LOGGER.debug("Sample already exists in dataset")
                return False
            
            df.loc[len(df)] = [message, label]
            df.to_csv(self.dataset_path, index=False)
            LOGGER.info(f"Added new sample to dataset: {message[:50]}... | label={label}")
            
            self._new_samples += 1
            return True
        except Exception as e:
            LOGGER.error(f"Failed to update dataset: {e}")
            return False
    
    def should_retrain(self, threshold: int = 100) -> bool:
        return self._new_samples >= threshold
