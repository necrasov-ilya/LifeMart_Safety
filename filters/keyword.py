from __future__ import annotations

import json
import re
from pathlib import Path

from core.types import FilterResult
from filters.base import BaseFilter
from utils.logger import get_logger

LOGGER = get_logger(__name__)


class KeywordFilter(BaseFilter):
    def __init__(self, keywords_path: Path | str | None = None):
        super().__init__("keyword")
        
        if keywords_path is None:
            keywords_path = Path(__file__).resolve().parents[1] / "data" / "keywords.json"
        
        self.keywords_path = Path(keywords_path)
        self.keywords: list[str] = []
        self.patterns: list[re.Pattern] = []
        self._load_keywords()
    
    def _load_keywords(self) -> None:
        if not self.keywords_path.exists():
            LOGGER.warning(f"Keywords file not found: {self.keywords_path}")
            self._create_default_keywords()
        
        try:
            with open(self.keywords_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.keywords = data.get("keywords", [])
                patterns = data.get("patterns", [])
                self.patterns = [re.compile(p, re.IGNORECASE) for p in patterns]
            LOGGER.info(f"Loaded {len(self.keywords)} keywords and {len(self.patterns)} patterns")
        except Exception as e:
            LOGGER.error(f"Failed to load keywords: {e}")
            self.keywords = []
            self.patterns = []
    
    def _create_default_keywords(self) -> None:
        default_data = {
            "keywords": [
                "заработ",
                "удалёнк",
                "удаленк",
                "казино",
                "ставк",
                "выигр",
                "доход",
                "работ на дому",
                "пассивный доход",
                "быстрые деньги",
                "инвестиц",
                "крипт",
                "трейд",
                "млм",
                "сетевой маркетинг"
            ],
            "patterns": [
                r"\$\d+",
                r"\d+\$",
                r"\d+\s*долларов",
                r"\d+\s*руб",
                r"https?://bit\.ly/",
                r"https?://t\.me/[a-zA-Z0-9_]+",
                r"\+\d{10,15}",
                r"пиши\s+[+«\"']",
                r"жми\s+сюда",
                r"переходи\s+по\s+ссылке"
            ]
        }
        
        self.keywords_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.keywords_path, "w", encoding="utf-8") as f:
            json.dump(default_data, f, ensure_ascii=False, indent=2)
        
        LOGGER.info(f"Created default keywords file: {self.keywords_path}")
        self.keywords = default_data["keywords"]
        self.patterns = [re.compile(p, re.IGNORECASE) for p in default_data["patterns"]]
    
    async def analyze(self, text: str) -> FilterResult:
        text_lower = text.lower()
        matched_keywords = []
        matched_patterns = []
        
        for keyword in self.keywords:
            if keyword.lower() in text_lower:
                matched_keywords.append(keyword)
        
        for pattern in self.patterns:
            if pattern.search(text):
                matched_patterns.append(pattern.pattern)
        
        total_matches = len(matched_keywords) + len(matched_patterns)
        
        if total_matches == 0:
            score = 0.0
        elif total_matches == 1:
            score = 0.4
        elif total_matches == 2:
            score = 0.6
        elif total_matches == 3:
            score = 0.8
        else:
            score = 0.95
        
        return FilterResult(
            filter_name=self.name,
            score=score,
            confidence=1.0,
            details={
                "matched_keywords": matched_keywords,
                "matched_patterns": matched_patterns,
                "total_matches": total_matches
            }
        )
    
    def is_ready(self) -> bool:
        return len(self.keywords) > 0 or len(self.patterns) > 0
