"""
services/meta_classifier.py
────────────────────────────────────────────────────────
Мета-классификатор на базе логистической регрессии с КОНТЕКСТОМ.

ОБНОВЛЕНО для контекстного анализа:
- Эмбеддинги: E_msg, E_ctx, E_user
- Центроиды: spam_centroid, ham_centroid
- Прототипы: семейства спама и легитимных сообщений
- Контекстные флаги: reply_to_staff, is_forwarded, author_is_admin, is_channel_announcement
- Whitelist анти-паттерны: store/order/brand

Артефакты загружаются из models/:
- meta_model.joblib (LogisticRegression или LightGBM)
- meta_calibrator.joblib (CalibratedClassifierCV)
- centroids.npz (spam_centroid, ham_centroid)
- prototypes.npz (семейства спама и легитимных сообщений)
- feature_spec.json (порядок и названия фичей)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import numpy as np

from core.types import AnalysisResult, EmbeddingVectors
from config.config import settings
from utils.logger import get_logger
from utils.textprep import extract_patterns_from_raw

LOGGER = get_logger(__name__)


# Названия семейств прототипов (должны совпадать с обучением)
SPAM_PROTOTYPES = [
    "recruiting",      # Рекрутинг (работа на дому, удалёнка)
    "gambling",        # Казино, ставки
    "loans",           # Займы, кредиты
    "get_rich",        # Быстрый заработок, инвестиции
    "dayjob",          # Подработка, халтура
    "illegal_docs",    # Документы, паспорта
    "gray_biz",        # Серый бизнес (дропы, курьеры)
]

LEGIT_PROTOTYPES = [
    "store_promo",     # Промо магазинов
    "giveaway_own",    # Розыгрыши своего канала
    "order_flow",      # Заказы, меню
    "service_info",    # Информация об услугах
]


class MetaClassifier:
    """
    Мета-классификатор для контекстного анализа спама.
    
    НОВЫЕ ФИЧИ:
    - sim_spam_msg, sim_ham_msg, delta_msg (для E_msg)
    - sim_spam_ctx, sim_ham_ctx, delta_ctx (для E_ctx)
    - sim_spam_user, sim_ham_user, delta_user (для E_user, опционально)
    - sim_proto_* для каждого прототипа
    - Контекстные флаги (reply_to_staff, is_forwarded, etc.)
    - Whitelist хиты (store, order, brand)
    """
    
    def __init__(self, models_dir: Path | str | None = None):
        if models_dir is None:
            models_dir = Path(__file__).resolve().parents[1] / "models"
        
        self.models_dir = Path(models_dir)
        
        # Модели
        self.logreg = None  # LogisticRegression или LightGBM
        self.calibrator = None
        
        # Центроиды
        self.spam_centroid = None
        self.ham_centroid = None
        
        # Прототипы
        self.spam_prototypes: Dict[str, np.ndarray] = {}
        self.legit_prototypes: Dict[str, np.ndarray] = {}
        
        # Спецификация фичей
        self.feature_names: List[str] = []
        
        self._load_artifacts()
    
    def _load_artifacts(self) -> None:
        """Загружает все артефакты мета-классификатора."""
        # Пути к файлам
        model_path = self.models_dir / settings.META_MODEL_PATH.split('/')[-1]
        calibrator_path = self.models_dir / settings.META_CALIBRATOR_PATH.split('/')[-1]
        centroids_path = self.models_dir / settings.CENTROIDS_PATH.split('/')[-1]
        prototypes_path = self.models_dir / settings.PROTOTYPES_PATH.split('/')[-1]
        feature_spec_path = self.models_dir / "feature_spec.json"
        
        if not model_path.exists():
            LOGGER.warning(
                f"Meta classifier not found at {model_path}. "
                "Run scripts/train_meta_context.py to train."
            )
            return
        
        try:
            from joblib import load
            
            # 1. Модель
            self.logreg = load(model_path)
            LOGGER.info(f"Loaded model from {model_path}")
            
            # 2. Калибратор
            if calibrator_path.exists():
                self.calibrator = load(calibrator_path)
                LOGGER.info(f"Loaded calibrator from {calibrator_path}")
            else:
                LOGGER.warning("Calibrator not found, predictions may be uncalibrated")
            
            # 3. Центроиды
            if centroids_path.exists():
                centroids_data = np.load(centroids_path)
                self.spam_centroid = centroids_data['spam_centroid']
                self.ham_centroid = centroids_data['ham_centroid']
                LOGGER.info(
                    f"Loaded centroids: spam={self.spam_centroid.shape}, "
                    f"ham={self.ham_centroid.shape}"
                )
            else:
                LOGGER.error("Centroids not found - required for classification")
                self.logreg = None
                return
            
            # 4. Прототипы
            if prototypes_path.exists():
                prototypes_data = np.load(prototypes_path)
                
                for proto_name in SPAM_PROTOTYPES:
                    key = f"spam_{proto_name}"
                    if key in prototypes_data:
                        self.spam_prototypes[proto_name] = prototypes_data[key]
                
                for proto_name in LEGIT_PROTOTYPES:
                    key = f"legit_{proto_name}"
                    if key in prototypes_data:
                        self.legit_prototypes[proto_name] = prototypes_data[key]
                
                LOGGER.info(
                    f"Loaded prototypes: {len(self.spam_prototypes)} spam, "
                    f"{len(self.legit_prototypes)} legit"
                )
            else:
                LOGGER.warning("Prototypes not found, will use only centroids")
            
            # 5. Feature spec
            if feature_spec_path.exists():
                with open(feature_spec_path, 'r', encoding='utf-8') as f:
                    spec = json.load(f)
                    self.feature_names = spec.get('features', [])
                LOGGER.info(f"Feature spec: {len(self.feature_names)} features")
            else:
                LOGGER.warning("Feature spec not found")
        
        except Exception as e:
            LOGGER.error(f"Failed to load artifacts: {e}", exc_info=True)
            self.logreg = None
    
    def is_ready(self) -> bool:
        """Проверяет готовность классификатора."""
        return (
            self.logreg is not None and
            self.calibrator is not None and
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
        analysis: AnalysisResult
    ) -> Tuple[np.ndarray, Dict]:
        """
        Строит ПОЛНЫЙ вектор фичей для контекстного мета-классификатора.
        
        ФИЧИ (в порядке feature_spec.json):
        1-3.   sim_spam_msg, sim_ham_msg, delta_msg
        4-6.   sim_spam_ctx, sim_ham_ctx, delta_ctx
        7-9.   sim_spam_user, sim_ham_user, delta_user (если есть E_user)
        10-16. sim_proto_recruiting, ...gambling, ...loans, ...get_rich, ...dayjob, ...illegal_docs, ...gray_biz
        17-20. sim_proto_store_promo, ...giveaway_own, ...order_flow, ...service_info
        21-22. kw_score, tfidf_score
        23-34. has_phone, has_url, has_email, has_money, money_count, has_age, 
               has_cta_plus, has_dm, has_remote, has_legal, has_casino, obfuscation_ratio
        35-38. reply_to_staff, is_forwarded, author_is_admin, is_channel_announcement
        39-41. whitelist_hits_store, whitelist_hits_order, whitelist_hits_brand
        
        Returns:
            (feature_vector, debug_info)
        """
        features = []
        debug = {}
        
        # Извлекаем эмбеддинги
        vectors: EmbeddingVectors = analysis.embedding_vectors or EmbeddingVectors()
        e_msg = np.array(vectors.E_msg) if vectors.E_msg else None
        e_ctx = np.array(vectors.E_ctx) if vectors.E_ctx else None
        e_user = np.array(vectors.E_user) if vectors.E_user else None
        
        # 1. Косинусы и дельты для E_msg
        if e_msg is not None and self.spam_centroid is not None:
            sim_spam_msg = self._cosine_similarity(e_msg, self.spam_centroid)
            sim_ham_msg = self._cosine_similarity(e_msg, self.ham_centroid)
            delta_msg = sim_spam_msg - sim_ham_msg
            
            features.extend([sim_spam_msg, sim_ham_msg, delta_msg])
            debug['sim_spam_msg'] = sim_spam_msg
            debug['sim_ham_msg'] = sim_ham_msg
            debug['delta_msg'] = delta_msg
        else:
            features.extend([0.0, 0.0, 0.0])
            debug['sim_spam_msg'] = None
            debug['sim_ham_msg'] = None
            debug['delta_msg'] = None
        
        # 2. Косинусы и дельты для E_ctx
        if e_ctx is not None and self.spam_centroid is not None:
            sim_spam_ctx = self._cosine_similarity(e_ctx, self.spam_centroid)
            sim_ham_ctx = self._cosine_similarity(e_ctx, self.ham_centroid)
            delta_ctx = sim_spam_ctx - sim_ham_ctx
            
            features.extend([sim_spam_ctx, sim_ham_ctx, delta_ctx])
            debug['sim_spam_ctx'] = sim_spam_ctx
            debug['sim_ham_ctx'] = sim_ham_ctx
            debug['delta_ctx'] = delta_ctx
        else:
            features.extend([0.0, 0.0, 0.0])
            debug['sim_spam_ctx'] = None
            debug['sim_ham_ctx'] = None
            debug['delta_ctx'] = None
        
        # 3. Косинусы и дельты для E_user (опционально)
        if e_user is not None and self.spam_centroid is not None:
            sim_spam_user = self._cosine_similarity(e_user, self.spam_centroid)
            sim_ham_user = self._cosine_similarity(e_user, self.ham_centroid)
            delta_user = sim_spam_user - sim_ham_user
            
            features.extend([sim_spam_user, sim_ham_user, delta_user])
            debug['sim_spam_user'] = sim_spam_user
            debug['sim_ham_user'] = sim_ham_user
            debug['delta_user'] = delta_user
        else:
            features.extend([0.0, 0.0, 0.0])
            debug['sim_spam_user'] = None
            debug['sim_ham_user'] = None
            debug['delta_user'] = None
        
        # 4. Прототипы SPAM
        proto_debug = {}
        for proto_name in SPAM_PROTOTYPES:
            if e_msg is not None and proto_name in self.spam_prototypes:
                sim = self._cosine_similarity(e_msg, self.spam_prototypes[proto_name])
                features.append(sim)
                proto_debug[f"sim_proto_{proto_name}"] = sim
            else:
                features.append(0.0)
                proto_debug[f"sim_proto_{proto_name}"] = None
        
        # 5. Прототипы LEGIT
        for proto_name in LEGIT_PROTOTYPES:
            if e_msg is not None and proto_name in self.legit_prototypes:
                sim = self._cosine_similarity(e_msg, self.legit_prototypes[proto_name])
                features.append(sim)
                proto_debug[f"sim_proto_{proto_name}"] = sim
            else:
                features.append(0.0)
                proto_debug[f"sim_proto_{proto_name}"] = None
        
        debug['prototypes'] = proto_debug
        
        # 6. Scores от старых фильтров (пониженный вес)
        kw_score = analysis.keyword_result.score
        tfidf_score = analysis.tfidf_result.score
        features.extend([kw_score, tfidf_score])
        debug['kw_score'] = kw_score
        debug['tfidf_score'] = tfidf_score
        
        # 7. Паттерн-фичи из сырого текста
        patterns = extract_patterns_from_raw(text)
        
        # Бинарные и численные паттерны
        features.append(float(patterns['has_phone']))
        features.append(float(patterns['has_url']))
        features.append(float(patterns['has_email']))
        features.append(float(patterns['has_money']))
        features.append(float(patterns['money_count']))
        features.append(float(patterns['has_age']))
        features.append(float(patterns['has_cta_plus']))
        features.append(float(patterns['has_dm']))
        features.append(float(patterns['has_remote']))
        features.append(float(patterns['has_legal']))
        features.append(float(patterns['has_casino']))
        features.append(float(patterns['obfuscation_ratio']))
        
        debug['patterns'] = patterns
        
        # 8. Контекстные флаги из метаданных
        metadata = analysis.metadata
        if metadata:
            features.append(float(metadata.reply_to_staff))
            features.append(float(metadata.is_forwarded))
            features.append(float(metadata.author_is_admin))
            features.append(float(metadata.is_channel_announcement))
            
            debug['context_flags'] = {
                'reply_to_staff': metadata.reply_to_staff,
                'is_forwarded': metadata.is_forwarded,
                'author_is_admin': metadata.author_is_admin,
                'is_channel_announcement': metadata.is_channel_announcement
            }
        else:
            features.extend([0.0, 0.0, 0.0, 0.0])
            debug['context_flags'] = None
        
        # 9. Whitelist хиты (анти-паттерны)
        whitelist_hits_store = patterns['whitelist_hits_store']
        whitelist_hits_order = patterns['whitelist_hits_order']
        whitelist_hits_brand = patterns['whitelist_hits_brand']
        
        features.append(float(whitelist_hits_store))
        features.append(float(whitelist_hits_order))
        features.append(float(whitelist_hits_brand))
        
        debug['whitelist_hits'] = {
            'store': whitelist_hits_store,
            'order': whitelist_hits_order,
            'brand': whitelist_hits_brand
        }
        
        # Собираем в numpy array
        X = np.array(features).reshape(1, -1)
        
        LOGGER.debug(f"Built feature vector: {X.shape}, {len(features)} features")
        
        return X, debug
    
    async def predict_proba(
        self,
        text: str,
        analysis: AnalysisResult
    ) -> Tuple[float, Dict]:
        """
        Вычисляет вероятность спама на основе ВСЕХ фичей.
        
        Args:
            text: исходный текст (для паттернов)
            analysis: результат анализа с эмбеддингами и метаданными
        
        Returns:
            (spam_probability, meta_debug_info)
        """
        if not self.is_ready():
            LOGGER.warning("Meta-classifier not ready, returning default 0.0")
            return 0.0, {'error': 'model_not_ready'}
        
        # Строим полный вектор фичей
        X, debug = self._build_features(text, analysis)
        
        # Предсказываем вероятность
        proba = self.calibrator.predict_proba(X)[0, 1]  # [0,1] - вероятность класса 1 (спам)
        
        # Добавляем топ-вклады фичей (если обучена LogReg)
        if hasattr(self.logreg, 'coef_'):
            top_contrib = self._compute_top_contributions(X[0], self.logreg.coef_[0])
            debug['top_features'] = top_contrib
        
        debug['p_spam'] = float(proba)
        debug['n_features'] = X.shape[1]
        
        LOGGER.debug(f"Meta-classifier: p_spam={proba:.3f}, n_features={X.shape[1]}")
        
        return float(proba), debug
    
    def _compute_top_contributions(
        self,
        feature_values: np.ndarray,
        coefficients: np.ndarray,
        top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Вычисляет топ-K фичей по вкладу в спам-скор (для отладки).
        
        Вклад = feature_value * coefficient
        """
        contributions = feature_values * coefficients
        
        # Берём индексы топ-K по абсолютному значению
        top_indices = np.argsort(np.abs(contributions))[::-1][:top_k]
        
        result = []
        for idx in top_indices:
            if idx < len(self.feature_names):
                fname = self.feature_names[idx]
                contrib = contributions[idx]
                result.append((fname, float(contrib)))
        
        return result
    
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
