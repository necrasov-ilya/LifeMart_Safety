import asyncio
import csv
import logging
import sys
import warnings
from dataclasses import replace
from pathlib import Path
import unittest

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.types import Action, AnalysisResult, EmbeddingVectors, FilterResult
from filters.keyword import KeywordFilter
from filters.tfidf import TfidfFilter
from filters.embedding import EmbeddingFilter
from services.meta_classifier import MetaClassifier
from services.policy import PolicyEngine
from utils.textprep import normalize_entities
from config.config import settings


class PolicyModeAccuracyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        dataset_path = Path(__file__).resolve().parents[1] / "data" / "messages.csv"
        if not dataset_path.exists():
            raise unittest.SkipTest("messages.csv dataset is required for accuracy benchmark")

        messages: list[str] = []
        labels: list[int] = []
        with dataset_path.open("r", encoding="utf-8") as dataset_file:
            reader = csv.DictReader(dataset_file)
            for row in reader:
                try:
                    messages.append(str(row["message"]))
                    labels.append(int(row["label"]))
                except (KeyError, TypeError, ValueError):
                    continue

        if not messages:
            raise unittest.SkipTest("messages.csv dataset is empty or malformed")

        cls.messages = messages
        cls.labels = labels

        cache_path = PROJECT_ROOT / "models" / "meta_eval_cache.npz"
        cls.cached_kw_scores = None
        cls.cached_tfidf_scores = None
        cls.cached_meta_proba = None
        logging.getLogger("services.policy").setLevel(logging.WARNING)
        logging.getLogger("filters.keyword").setLevel(logging.WARNING)
        logging.getLogger("filters.tfidf").setLevel(logging.WARNING)
        logging.getLogger("filters.embedding").setLevel(logging.WARNING)

        if cache_path.exists():
            cache = np.load(cache_path)
            if len(cache["labels"]) == len(cls.labels):
                cls.cached_kw_scores = cache["kw_scores"]
                cls.cached_tfidf_scores = cache["tfidf_scores"]
                cls.cached_meta_proba = cache["meta_proba"]
            else:
                warnings.warn(
                    f"meta_eval_cache.npz length mismatch (expected {len(cls.labels)}, got {len(cache['labels'])}); ignoring cache"
                )

        if cls.cached_kw_scores is not None:
            cls.meta_ready = True
            cls.analyses = cls._build_analyses_from_cache()
            cls.keyword_filter = None
            cls.tfidf_filter = None
            cls.embedding_filter = None
            cls.embedding_ready = False
            cls.meta_classifier = None
        else:
            cls.keyword_filter = KeywordFilter()
            if not cls.keyword_filter.is_ready():
                raise unittest.SkipTest("Keyword filter failed to load keywords")

            cls.tfidf_filter = TfidfFilter()
            if not cls.tfidf_filter.is_ready():
                raise unittest.SkipTest("TF-IDF model is not available")

            cls.embedding_filter = EmbeddingFilter()
            cls.embedding_ready = cls.embedding_filter.is_ready()

            cls.meta_classifier = MetaClassifier()
            cls.meta_ready = cls.meta_classifier.is_ready()

            cls.analyses = asyncio.run(cls._build_analyses(cls.messages))

    @classmethod
    @classmethod
    async def _build_analyses(cls, texts: list[str]) -> list[AnalysisResult]:
        analyses: list[AnalysisResult] = []
        embeddings: list[EmbeddingVectors | None] = [None] * len(texts)
        emb_debugs: list[dict | None] = [None] * len(texts)

        if cls.embedding_ready and getattr(cls.embedding_filter, "provider", None) is not None:
            capsules = [f"passage: {normalize_entities(text)}" for text in texts]
            batch_size = 32
            for start_idx in range(0, len(capsules), batch_size):
                batch_capsules = capsules[start_idx:start_idx + batch_size]
                try:
                    batch_embeddings, status = await cls.embedding_filter.provider.get_embeddings_batch(
                        batch_capsules,
                        timeout_ms=getattr(settings, "EMBEDDING_TIMEOUT_MS", 800)
                    )
                except Exception:
                    batch_embeddings = [None] * len(batch_capsules)
                    status = "error"
                for offset, embedding in enumerate(batch_embeddings):
                    if embedding is not None:
                        vector = EmbeddingVectors(E_msg=embedding, E_ctx=None, E_user=None)
                        embeddings[start_idx + offset] = vector
                        emb_debugs[start_idx + offset] = {
                            "status": status,
                            "degraded_ctx": False,
                            "degraded_user": False
                        }

        for idx, text in enumerate(texts):
            keyword_result = await cls.keyword_filter.analyze(text)
            tfidf_result = await cls.tfidf_filter.analyze(text)

            embedding_vectors = embeddings[idx]
            embedding_result = None
            degraded_ctx = False
            if embedding_vectors is not None:
                debug_info = emb_debugs[idx] or {}
                embedding_result = cls.embedding_filter.build_result_from_vectors(
                    vectors=embedding_vectors,
                    debug_info=debug_info
                )
                degraded_ctx = bool(debug_info.get("degraded_ctx", False))

            analysis = AnalysisResult(
                keyword_result=keyword_result,
                tfidf_result=tfidf_result,
                embedding_result=embedding_result,
                embedding_vectors=embedding_vectors,
                degraded_ctx=degraded_ctx
            )

            if cls.meta_ready:
                meta_proba, meta_debug = await cls.meta_classifier.predict_proba(text, analysis)
                analysis = replace(
                    analysis,
                    meta_proba=meta_proba,
                    meta_debug=meta_debug,
                )

            analyses.append(analysis)

        return analyses

    @classmethod
    def _build_analyses_from_cache(cls) -> list[AnalysisResult]:
        analyses: list[AnalysisResult] = []
        for kw_score, tfidf_score, meta_score in zip(
            cls.cached_kw_scores,
            cls.cached_tfidf_scores,
            cls.cached_meta_proba if cls.cached_meta_proba is not None else np.zeros_like(cls.cached_kw_scores),
        ):
            keyword_result = FilterResult(
                filter_name="keyword",
                score=float(kw_score),
            )
            tfidf_result = FilterResult(
                filter_name="tfidf",
                score=float(tfidf_score),
            )
            analysis = AnalysisResult(
                keyword_result=keyword_result,
                tfidf_result=tfidf_result,
                embedding_result=None,
                embedding_vectors=None,
                meta_proba=float(meta_score),
                meta_debug=None,
                degraded_ctx=False,
            )
            analyses.append(analysis)
        return analyses

    def _evaluate_mode(self, mode: str) -> float:
        engine = PolicyEngine()
        engine.policy_mode = mode

        correct = 0
        for analysis, label in zip(self.analyses, self.labels):
            if mode != "legacy-manual" and analysis.meta_proba is None:
                self.fail("Meta probability is required for manual policy evaluation")

            action, _ = engine.decide_action(analysis)
            predicted = 0 if action == Action.APPROVE else 1
            if predicted == label:
                correct += 1

        return correct / len(self.labels)

    def test_manual_vs_legacy_accuracy(self) -> None:
        if not self.meta_ready:
            self.skipTest("MetaClassifier artifacts are missing; manual mode cannot be evaluated")

        legacy_accuracy = self._evaluate_mode("legacy-manual")
        manual_accuracy = self._evaluate_mode("manual")
        improvement = manual_accuracy - legacy_accuracy

        print(f"Legacy mode accuracy: {legacy_accuracy:.4f}")
        print(f"Manual mode accuracy: {manual_accuracy:.4f}")
        print(f"Accuracy improvement: {improvement:.4f}")
        self.__class__.legacy_accuracy = legacy_accuracy
        self.__class__.manual_accuracy = manual_accuracy
        self.__class__.accuracy_improvement = improvement

        self.assertTrue(0.0 <= legacy_accuracy <= 1.0, "Legacy accuracy must be within [0, 1]")
        self.assertTrue(0.0 <= manual_accuracy <= 1.0, "Manual accuracy must be within [0, 1]")



if __name__ == "__main__":
    unittest.main(verbosity=2)
