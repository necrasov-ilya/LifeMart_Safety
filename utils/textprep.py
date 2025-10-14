"""
utils/textprep.py
────────────────────────────────────────────────────────
Подготовка текста для эмбеддинга с нормализацией и заменой на плейсхолдеры.

• NFKC нормализация Unicode
• Замены конфиденциальных/шумных паттернов на стабильные токены
• Префикс 'passage: ' для multilingual-e5-small
• Исходный текст не модифицируется (для TF-IDF/Keyword)
"""

from __future__ import annotations

import re
import unicodedata


def prepare_for_embedding(text: str) -> str:
    """
    Нормализует текст для получения эмбеддинга.
    
    Применяет:
    - NFKC нормализацию
    - Замены URL, телефонов, email, денег, возраста, TG-ссылок на плейсхолдеры
    - Схлопывание пробелов
    - Префикс 'passage: '
    
    Args:
        text: исходный текст сообщения
        
    Returns:
        нормализованный текст с префиксом 'passage: '
    """
    # NFKC нормализация Unicode
    normalized = unicodedata.normalize('NFKC', text)
    
    # URL (должен быть раньше TG, т.к. t.me - это тоже URL)
    # https://example.com, http://bit.ly/xxx, www.site.ru
    normalized = re.sub(
        r'https?://[^\s]+|www\.[^\s]+',
        '<URL>',
        normalized,
        flags=re.IGNORECASE
    )
    
    # Telegram ссылки и хэндлы (@username, t.me/username)
    # Должен быть после URL, чтобы t.me/xxx не заменилось дважды
    normalized = re.sub(
        r't\.me/[a-zA-Z0-9_]+',
        '<TG>',
        normalized,
        flags=re.IGNORECASE
    )
    normalized = re.sub(
        r'@[a-zA-Z0-9_]{3,}',
        '<TG>',
        normalized
    )
    
    # Телефоны: +7(XXX)XXX-XX-XX, 8XXXXXXXXXX, +1234567890, и вариации
    normalized = re.sub(
        r'(?:\+\d{1,3})?[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{2}[-.\s]?\d{2}',
        '<PHONE>',
        normalized
    )
    normalized = re.sub(
        r'\+\d{10,15}',
        '<PHONE>',
        normalized
    )
    
    # Email
    normalized = re.sub(
        r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
        '<EMAIL>',
        normalized,
        flags=re.IGNORECASE
    )
    
    # Деньги и валюты
    # Примеры: 1000руб, 50 рублей, 5000р, $100, 20$, 15к, 30 тыс, 10%, USDT, Bitcoin
    money_patterns = [
        r'\d+\s*(?:руб(?:л(?:ей|я|ь)?)?|р\.?|₽)',
        r'(?:\$|USD|usd)\s*\d+',
        r'\d+\s*(?:\$|USD|usd)',
        r'\d+\s*(?:к|тыс\.?|тысяч)',
        r'\d+\s*%',
        r'(?:доллар|евро|usdt|btc|bitcoin|ethereum)',
    ]
    for pattern in money_patterns:
        normalized = re.sub(pattern, '<MONEY>', normalized, flags=re.IGNORECASE)
    
    # Возрастные ограничения: 18+, 21+, 23+, 18 лет, от 18
    normalized = re.sub(
        r'(?:от\s+)?\d{2}\+|(?:от\s+)?\d{2}\s*(?:лет|года?)',
        '<AGE>',
        normalized,
        flags=re.IGNORECASE
    )
    
    # Схлопывание множественных пробелов
    normalized = re.sub(r'\s+', ' ', normalized).strip()
    
    # Префикс для multilingual-e5-small
    return f"passage: {normalized}"


def extract_patterns_from_raw(text: str) -> dict[str, any]:
    """
    Извлекает паттерны из сырого текста (без нормализации).
    Используется для построения фичей в мета-классификаторе.
    
    Returns:
        dict с ключами:
        - has_money, money_count
        - has_age
        - has_cta_plus
        - has_dm
        - has_contact
        - has_remote
        - has_legal
        - has_casino
        - obfuscation_ratio
    """
    text_lower = text.lower()
    
    # Деньги
    money_matches = re.findall(
        r'\d+\s*(?:руб|р\.?|₽|\$|usd|к|тыс|евро|usdt|btc|%)',
        text_lower
    )
    has_money = len(money_matches) > 0
    money_count = len(money_matches)
    
    # Возраст
    has_age = bool(re.search(r'(?:от\s+)?\d{2}\+|(?:от\s+)?\d{2}\s*(?:лет|года?)', text_lower))
    
    # CTA с плюсом/стартом
    cta_patterns = [
        r'пиш[иу](?:те)?\s+[+«"\']?\+',
        r'жми\s+(?:на\s+)?[+«"\']',
        r'старт',
        r'пиш[иу](?:те)?\s+в\s+(?:лс|личку)',
    ]
    has_cta_plus = any(re.search(p, text_lower) for p in cta_patterns)
    
    # Призыв к ЛС/личке
    has_dm = bool(re.search(r'\b(?:лс|личк[аеу]|в\s+личные)\b', text_lower))
    
    # Контакты (URL, TG, телефон, email)
    has_contact = bool(
        re.search(r'https?://|www\.|t\.me/|@[a-zA-Z0-9_]{3,}', text) or
        re.search(r'\+\d{10,15}|[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', text)
    )
    
    # Удалённая работа
    has_remote = bool(re.search(r'удал[её]нк|удал[её]нн|дистанц', text_lower))
    
    # Легальность/безопасность (часто в спаме для "прикрытия")
    legal_patterns = [
        r'легальн',
        r'честн',
        r'без\s+риск',
        r'без\s+вложен',
        r'не\s+нарко',
        r'не\s+мошен',
    ]
    has_legal = any(re.search(p, text_lower) for p in legal_patterns)
    
    # Казино/ставки
    casino_patterns = [
        r'казино|казик',
        r'фриспин',
        r'ставк[аи]',
        r'rtp',
        r'букмекер',
        r'слот',
        r'выигр',
    ]
    has_casino = any(re.search(p, text_lower) for p in casino_patterns)
    
    # Обфускация: смешанные кириллица/латиница, цифроподмены
    # Примеры: зeлёный (е латинское), 3амена (3 вместо з)
    total_chars = len(text)
    if total_chars == 0:
        obfuscation_ratio = 0.0
    else:
        # Подсчёт смешанных кир/лат в словах
        words = re.findall(r'\b\w+\b', text)
        obfuscated_words = 0
        for word in words:
            has_cyrillic = bool(re.search(r'[а-яА-ЯёЁ]', word))
            has_latin = bool(re.search(r'[a-zA-Z]', word))
            has_digit_in_letters = bool(re.search(r'[а-яА-ЯёЁa-zA-Z]+\d+[а-яА-ЯёЁa-zA-Z]+', word))
            
            if (has_cyrillic and has_latin) or has_digit_in_letters:
                obfuscated_words += 1
        
        obfuscation_ratio = obfuscated_words / max(len(words), 1)
    
    return {
        'has_money': has_money,
        'money_count': money_count,
        'has_age': has_age,
        'has_cta_plus': has_cta_plus,
        'has_dm': has_dm,
        'has_contact': has_contact,
        'has_remote': has_remote,
        'has_legal': has_legal,
        'has_casino': has_casino,
        'obfuscation_ratio': obfuscation_ratio,
    }
