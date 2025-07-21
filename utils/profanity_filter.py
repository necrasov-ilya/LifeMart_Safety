# utils/profanity_filter.py
"""
Антимат фильтр для обнаружения нецензурной лексики в сообщениях.
"""
import re
from pathlib import Path
from typing import List, Set
from utils.logger import get_logger

LOGGER = get_logger(__name__)

class ProfanityFilter:
    """Фильтр для обнаружения нецензурной лексики."""

    def __init__(self, profanity_file: str = "data/profanity_russian.txt"):
        self.profanity_file = Path(profanity_file)
        self.profanity_words: Set[str] = set()
        self.load_profanity_words()

    def load_profanity_words(self) -> None:
        """Загружает список запрещенных слов из файла."""
        try:
            if self.profanity_file.exists():
                with open(self.profanity_file, 'r', encoding='utf-8') as f:
                    self.profanity_words = {
                        word.strip().lower()
                        for word in f.readlines()
                        if word.strip()
                    }
                LOGGER.info(f"Загружено {len(self.profanity_words)} запрещенных слов")
            else:
                LOGGER.warning(f"Файл с матом не найден: {self.profanity_file}")
        except Exception as e:
            LOGGER.error(f"Ошибка загрузки файла с матом: {e}")

    def _normalize_text(self, text: str) -> str:
        """Нормализует текст для проверки."""
        # Убираем символы-разделители, которые могут маскировать мат
        text = re.sub(r'[^\w\s]', '', text.lower())
        # Убираем повторяющиеся символы (пыыыздец -> пыздец)
        text = re.sub(r'(.)\1{2,}', r'\1\1', text)
        # Убираем пробелы
        text = re.sub(r'\s+', '', text)
        return text

    def contains_profanity(self, text: str) -> bool:
        """Проверяет, содержит ли текст нецензурную лексику."""
        if not text or not self.profanity_words:
            return False

        normalized = self._normalize_text(text)

        # Прямая проверка
        for word in self.profanity_words:
            if word in normalized:
                return True

        # Проверка с учетом замен символов
        substitutions = {
            '0': 'о', '3': 'з', '4': 'ч', '6': 'б', '9': 'д',
            '@': 'а', '$': 's', '!': 'i', '1': 'l', '7': 'т'
        }

        for old, new in substitutions.items():
            if old in normalized:
                modified = normalized.replace(old, new)
                for word in self.profanity_words:
                    if word in modified:
                        return True

        return False

    def get_profanity_words(self, text: str) -> List[str]:
        """Возвращает список найденных запрещенных слов."""
        if not text or not self.profanity_words:
            return []

        normalized = self._normalize_text(text)
        found_words = []

        for word in self.profanity_words:
            if word in normalized:
                found_words.append(word)

        return found_words

    def add_word(self, word: str) -> bool:
        """Добавляет новое слово в список запрещенных."""
        word = word.strip().lower()
        if word and word not in self.profanity_words:
            self.profanity_words.add(word)
            self._save_to_file()
            LOGGER.info(f"Добавлено новое запрещенное слово: {word}")
            return True
        return False

    def remove_word(self, word: str) -> bool:
        """Удаляет слово из списка запрещенных."""
        word = word.strip().lower()
        if word in self.profanity_words:
            self.profanity_words.remove(word)
            self._save_to_file()
            LOGGER.info(f"Удалено запрещенное слово: {word}")
            return True
        return False

    def _save_to_file(self) -> None:
        """Сохраняет список запрещенных слов в файл."""
        try:
            with open(self.profanity_file, 'w', encoding='utf-8') as f:
                for word in sorted(self.profanity_words):
                    f.write(f"{word}\n")
        except Exception as e:
            LOGGER.error(f"Ошибка сохранения файла с матом: {e}")

# Глобальный экземпляр фильтра
profanity_filter = ProfanityFilter()
