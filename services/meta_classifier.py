"""
services/meta_classifier.py
────────────────────────────────────────────────────────
Мета-классификатор на базе логистической регрессии.

Принимает фичи от фильтров + эмбеддинг + паттерны → возвращает калиброванную p_spam.
Артефакты загружаются из models/:
- meta_logreg.joblib (LogisticRegression)
- meta_calibrator.joblib (CalibratedClassifierCV или None)
- centroids.npz (spam_centroid, ham_centroid)
- feature_spec.json (порядок и названия фичей)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np

from core.types import AnalysisResult
from utils.logger import get_logger
from utils.textprep import extract_patterns_from_raw

LOGGER = get_logger(__name__)


class MetaClassifier:
    """
    Мета-классификатор для принятия финального решения о спаме.
    
    Использует:
    - Эмбеддинги + косинусные дистанции до центроидов
    - Scores от Keyword/TF-IDF фильтров
    - Паттерн-фичи (money, age, cta, etc.)
    """
    
    def __init__(self, models_dir: Path | str | None = None):
        if models_dir is None:
            models_dir = Path(__file__).resolve().parents[1] / "models"
        
        self.models_dir = Path(models_dir)
        
        self.logreg = None
        self.calibrator = None
        self.spam_centroid = None
        self.ham_centroid = None
        self.feature_names = []
        
        self._load_artifacts()
    
    def _load_artifacts(self) -> None:
        """Загружает артефакты обученной модели."""
        logreg_path = self.models_dir / "meta_logreg.joblib"
        calibrator_path = self.models_dir / "meta_calibrator.joblib"
        centroids_path = self.models_dir / "centroids.npz"
        feature_spec_path = self.models_dir / "feature_spec.json"
        
        if not logreg_path.exists():
            LOGGER.warning(
                f"Meta classifier artifacts not found at {self.models_dir}. "
                "Run scripts/train_meta.py to train the model."
            )
            return
        
        try:
            from joblib import load
            
            self.logreg = load(logreg_path)
            LOGGER.info(f"Loaded LogisticRegression from {logreg_path}")
            
            if calibrator_path.exists():
                self.calibrator = load(calibrator_path)
                LOGGER.info(f"Loaded calibrator from {calibrator_path}")
            
            if centroids_path.exists():
                centroids_data = np.load(centroids_path)
                self.spam_centroid = centroids_data['spam_centroid']
                self.ham_centroid = centroids_data['ham_centroid']
                LOGGER.info(
                    f"Loaded centroids: spam {self.spam_centroid.shape}, "
                    f"ham {self.ham_centroid.shape}"
                )
            
            if feature_spec_path.exists():
                with open(feature_spec_path, 'r') as f:
                    spec = json.load(f)
                    self.feature_names = spec.get('features', [])
                LOGGER.info(f"Feature spec: {len(self.feature_names)} features")
        
        except Exception as e:
            LOGGER.error(f"Failed to load meta classifier artifacts: {e}")
            self.logreg = None
    
    def is_ready(self) -> bool:
        """Проверяет готовность классификатора."""
        return (
            self.logreg is not None and
            self.spam_centroid is not None and
            self.ham_centroid is not None and
            len(self.feature_names) > 0
        )
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Вычисляет косинусное сходство между двумя векторами."""
        dot = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(dot / (norm1 * norm2))
    
    def _build_features(
        self,
        text: str,
        embedding: Optional[np.ndarray],
        analysis: AnalysisResult
    ) -> tuple[np.ndarray, dict]:
        """
        Строит вектор фичей для классификации.
        
        Returns:
            (feature_vector, debug_info)
        """
        features = []
        debug = {}
        
        # 1. Косинусные дистанции до центроидов (если есть эмбеддинг)
        if embedding is not None and self.spam_centroid is not None and self.ham_centroid is not None:
            sim_spam = self._cosine_similarity(embedding, self.spam_centroid)
            sim_ham = self._cosine_similarity(embedding, self.ham_centroid)
            sim_diff = sim_spam - sim_ham
            
            features.extend([sim_spam, sim_ham, sim_diff])
            debug['sim_spam'] = sim_spam
            debug['sim_ham'] = sim_ham
            debug['sim_diff'] = sim_diff
        else:
            # Если нет эмбеддинга - заполняем нулями
            features.extend([0.0, 0.0, 0.0])
            debug['sim_spam'] = None
            debug['sim_ham'] = None
            debug['sim_diff'] = None
        
        # 2. Scores от фильтров
        kw_score = analysis.keyword_result.score
        tfidf_score = analysis.tfidf_result.score
        
        features.extend([kw_score, tfidf_score])
        debug['kw_score'] = kw_score
        debug['tfidf_score'] = tfidf_score
        
        # 3. Паттерны из сырого текста
        patterns = extract_patterns_from_raw(text)
        
        # Бинарные фичи (float для sklearn)
        features.append(float(patterns['has_money']))
        features.append(float(patterns['money_count']))
        features.append(float(patterns['has_age']))
        features.append(float(patterns['has_cta_plus']))
        features.append(float(patterns['has_dm']))
        features.append(float(patterns['has_contact']))
        features.append(float(patterns['has_remote']))
        features.append(float(patterns['has_legal']))
        features.append(float(patterns['has_casino']))
        features.append(patterns['obfuscation_ratio'])
        
        debug['patterns'] = patterns
        
        return np.array(features).reshape(1, -1), debug
    
    async def predict_proba(
        self,
        text: str,
        analysis: AnalysisResult
    ) -> tuple[float, dict]:
        """
        Предсказывает вероятность спама.
        
        Args:
            text: исходный текст сообщения
            analysis: результат анализа фильтров
            
        Returns:
            (p_spam, debug_info)
            p_spam ∈ [0, 1] - калиброванная вероятность
            debug_info - словарь с деталями для логирования/UI
        """
        if not self.is_ready():
            LOGGER.warning("Meta classifier not ready, returning None")
            return None, {"error": "Meta classifier not trained"}
        
        # Извлекаем эмбеддинг из результата EmbeddingFilter
        embedding = None
        if analysis.embedding_result and analysis.embedding_result.details:
            emb_data = analysis.embedding_result.details.get('embedding')
            if emb_data is not None:
                embedding = np.array(emb_data)
        
        # Строим фичи
        X, debug = self._build_features(text, embedding, analysis)
        
        # Предсказание
        try:
            if self.calibrator:
                # Калиброванная вероятность
                proba = self.calibrator.predict_proba(X)[0, 1]
                debug['calibrated'] = True
            else:
                # Некалиброванная вероятность от LogisticRegression
                proba = self.logreg.predict_proba(X)[0, 1]
                debug['calibrated'] = False
            
            p_spam = float(proba)
            debug['p_spam'] = p_spam
            
            LOGGER.debug(
                f"Meta prediction: p_spam={p_spam:.3f}, "
                f"sim_diff={debug.get('sim_diff', 'N/A')}, "
                f"patterns={sum(1 for k, v in debug['patterns'].items() if v and k.startswith('has_'))}"
            )
            
            return p_spam, debug
        
        except Exception as e:
            LOGGER.error(f"Meta classifier prediction failed: {e}")
            return None, {"error": str(e)}
    
    def get_info(self) -> dict:
        """Возвращает информацию о загруженных артефактах."""
        info = {
            "ready": self.is_ready(),
            "models_dir": str(self.models_dir),
            "logreg_loaded": self.logreg is not None,
            "calibrator_loaded": self.calibrator is not None,
            "centroids_loaded": self.spam_centroid is not None and self.ham_centroid is not None,
            "num_features": len(self.feature_names),
            "feature_names": self.feature_names,
        }
        
        # Даты модификации файлов
        try:
            logreg_path = self.models_dir / "meta_logreg.joblib"
            if logreg_path.exists():
                import datetime
                mtime = logreg_path.stat().st_mtime
                info['logreg_date'] = datetime.datetime.fromtimestamp(mtime).isoformat()
        except Exception:
            pass
        
        return info
