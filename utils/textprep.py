"""
utils/textprep.py
────────────────────────────────────────────────────────
Подготовка текста для контекстного анализа с эмбеддингами.

НОВАЯ ВЕРСИЯ:
• normalize_entities() - стабильные теги для сущностей
• build_context_capsule() - формирование контекстной капсулы
• build_user_capsule() - капсула истории пользователя
• Whitelist-словари (STORE_TERMS, ORDER_TERMS, BRAND_TERMS)
• Лимиты по токенам (~512 для контекста)
"""

from __future__ import annotations

import re
import unicodedata
from typing import List, Dict, Tuple


# WHITELIST СЛОВАРИ (анти-паттерны для снижения p_spam)

STORE_TERMS = {
    # Магазин/продажи
    "магазин", "магаз", "товар", "продук", "ассортимент", "наличи", 
    "склад", "поступление", "новинк", "коллекц", "каталог",
    "цена", "стоимост", "прайс", "скидк", "акци", "распродаж",
    "купить", "заказ", "доставк", "самовывоз", "оплат",
    "размер", "цвет", "модел", "бренд", "производител",
    # Онлайн торговля
    "интернет-магазин", "сайт", "корзин", "оформ", "чек",
    "гарантия", "возврат", "обмен", "качеств",
}

ORDER_TERMS = {
    # Заказы/меню
    "заказ", "меню", "блюд", "порци", "доставка еды", "ресторан", "кафе",
    "пицц", "суши", "ролл", "бургер", "салат", "напиток",
    "завтрак", "обед", "ужин", "комбо", "сет",
    "адрес доставки", "время доставки", "курьер", "оплата при получении",
    # Услуги
    "услуг", "запись", "сеанс", "мастер", "специалист", "консультац",
    "расписани", "график работы", "прием", "бронирован",
}

BRAND_TERMS = {
    "lifemart", "life mart", "life-mart", "lifemart.ru",
    "жизньмарт", "жизнь март", "лайфмарт", "лайф март",
    # Ресторанный бренд «Сушкоф» (варианты написания)
    "сушкоф", "сушкофф", "sushkof", "sushkoff", "sushkof.ru",
}

# Компиляция в нижний регистр для быстрого поиска
_STORE_TERMS_LOWER = {term.lower() for term in STORE_TERMS}
_ORDER_TERMS_LOWER = {term.lower() for term in ORDER_TERMS}
_BRAND_TERMS_LOWER = {term.lower() for term in BRAND_TERMS}


# НОРМАЛИЗАЦИЯ СУЩНОСТЕЙ

def normalize_entities(text: str) -> str:
    """
    Нормализует текст со стабильными тегами для сущностей.
    
    Порядок замен критичен (URL → TG → PHONE → EMAIL → MONEY → AGE).
    
    Теги:
        <URL> - http(s)://..., www...
        <TG> - t.me/username, @username
        <PHONE> - телефоны (различные форматы)
        <EMAIL> - email адреса
        <MONEY> - деньги, валюты, проценты
        <AGE> - возрастные ограничения (18+, 21+, от 18 лет)
    
    Args:
        text: сырой текст сообщения
        
    Returns:
        текст с замененными сущностями
    """
    # NFKC нормализация Unicode
    normalized = unicodedata.normalize('NFKC', text)
    
    # 1. URL (ПЕРЕД TG, т.к. t.me - тоже URL)
    # Формат: https://example.com, http://bit.ly/xxx, www.site.ru
    normalized = re.sub(
        r'https?://[^\s]+|www\.[^\s]+',
        '<URL>',
        normalized,
        flags=re.IGNORECASE
    )
    
    # 2. Telegram ссылки и хэндлы
    # t.me/username (если не заменилось как URL)
    normalized = re.sub(
        r't\.me/[a-zA-Z0-9_]+',
        '<TG>',
        normalized,
        flags=re.IGNORECASE
    )
    # @username (минимум 3 символа)
    normalized = re.sub(
        r'@[a-zA-Z0-9_]{3,}',
        '<TG>',
        normalized
    )
    
    # 3. Телефоны
    # Формат: +7(XXX)XXX-XX-XX, 8XXXXXXXXXX, +1234567890, и вариации
    # Сначала полные форматы с разделителями
    normalized = re.sub(
        r'(?:\+\d{1,3})?[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{2}[-.\s]?\d{2}',
        '<PHONE>',
        normalized
    )
    # Затем простые +XXXXXXXXXXXX
    normalized = re.sub(
        r'\+\d{10,15}',
        '<PHONE>',
        normalized
    )
    
    # 4. Email
    normalized = re.sub(
        r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
        '<EMAIL>',
        normalized,
        flags=re.IGNORECASE
    )
    
    # 5. Деньги и валюты
    # Порядок важен: сначала сложные конструкции, потом простые
    money_patterns = [
        # Число + рубли (1000руб, 50 рублей, 5000р, 100₽)
        r'\d+\s*(?:руб(?:л(?:ей|я|ь)?)?|р\.?|₽)',
        # Доллары ($100, USD 100, 100 usd)
        r'(?:\$|USD|usd)\s*\d+',
        r'\d+\s*(?:\$|USD|usd)',
        # Тысячи (15к, 30 тыс., 20 тысяч)
        r'\d+\s*(?:к|тыс\.?|тысяч)',
        # Проценты (10%, 20 процентов)
        r'\d+\s*(?:%|процент)',
        # Криптовалюты и валюты
        r'(?:доллар|евро|usdt|btc|bitcoin|eth|ethereum)',
    ]
    for pattern in money_patterns:
        normalized = re.sub(pattern, '<MONEY>', normalized, flags=re.IGNORECASE)
    
    # 6. Возрастные ограничения
    # 18+, 21+, 23+, 18 лет, от 18, 18 года
    normalized = re.sub(
        r'(?:от\s+)?\d{2}\+|(?:от\s+)?\d{2}\s*(?:лет|года?)',
        '<AGE>',
        normalized,
        flags=re.IGNORECASE
    )
    
    # Схлопывание множественных пробелов
    normalized = re.sub(r'\s+', ' ', normalized).strip()
    
    return normalized


# ПОСТРОЕНИЕ КАПСУЛ

def build_context_capsule(
    message: str,
    history: List[str],
    metadata: Dict[str, any],
    max_chars: int = 2048  # ~512 токенов для e5-small
) -> str:
    """
    Строит контекстную капсулу с метаданными и историей.
    
    Формат:
        passage:
        [CTX:n=<N>][reply_to_staff=0/1][is_forwarded=0/1][author_is_admin=0/1][is_channel_announcement=0/1]
        <normalized current message>
        ---
        <normalized prev1>
        ---
        <normalized prev2>
        ...
    
    Усечение происходит с конца истории (самые старые сообщения).
    
    Args:
        message: текущее сообщение (сырое)
        history: список предыдущих сообщений (от новых к старым)
        metadata: словарь с ключами reply_to_staff, is_forwarded, author_is_admin, is_channel_announcement
        max_chars: максимальная длина капсулы в символах
        
    Returns:
        отформатированная капсула с префиксом "passage:"
    """
    # Нормализация текущего сообщения
    norm_message = normalize_entities(message)
    
    # Формирование метаданных (бинарные флаги)
    reply_to_staff = int(metadata.get('reply_to_staff', False))
    is_forwarded = int(metadata.get('is_forwarded', False))
    author_is_admin = int(metadata.get('author_is_admin', False))
    is_channel_announcement = int(metadata.get('is_channel_announcement', False))
    
    n_history = len(history)
    
    # Заголовок капсулы
    header = (
        f"passage:\n"
        f"[CTX:n={n_history}]"
        f"[reply_to_staff={reply_to_staff}]"
        f"[is_forwarded={is_forwarded}]"
        f"[author_is_admin={author_is_admin}]"
        f"[is_channel_announcement={is_channel_announcement}]\n"
        f"{norm_message}"
    )
    
    # Бюджет на историю
    remaining = max_chars - len(header)
    
    if remaining <= 0:
        # Сообщение слишком длинное, усекаем его
        truncated_message = norm_message[:max_chars - 200]  # Оставляем место для header
        return (
            f"passage:\n"
            f"[CTX:n=0]"
            f"[reply_to_staff={reply_to_staff}]"
            f"[is_forwarded={is_forwarded}]"
            f"[author_is_admin={author_is_admin}]"
            f"[is_channel_announcement={is_channel_announcement}]\n"
            f"{truncated_message}"
        )
    
    # Добавление истории (с конца, усекаем самые старые)
    history_parts = []
    for msg in history:
        norm_hist = normalize_entities(msg)
        separator = "\n---\n"
        
        if remaining < len(separator) + len(norm_hist):
            # Бюджет исчерпан, останавливаемся
            break
        
        history_parts.append(norm_hist)
        remaining -= len(separator) + len(norm_hist)
    
    # Сборка финальной капсулы
    if history_parts:
        capsule = header + "\n---\n" + "\n---\n".join(history_parts)
    else:
        capsule = header
    
    return capsule


def build_user_capsule(
    last_k_msgs: List[str],
    max_chars: int = 512  # Компактная капсула
) -> str:
    """
    Строит капсулу истории пользователя (краткая сводка).
    
    Формат:
        passage:
        [USER:k=<K>]
        <msg1>
        ---
        <msg2>
        ...
    
    Каждое сообщение усекается до 100 символов для компактности.
    
    Args:
        last_k_msgs: последние K сообщений пользователя (от новых к старым)
        max_chars: максимальная длина капсулы
        
    Returns:
        отформатированная user-капсула
    """
    if not last_k_msgs:
        return "passage:\n[USER:k=0]\n"
    
    k = len(last_k_msgs)
    header = f"passage:\n[USER:k={k}]\n"
    
    remaining = max_chars - len(header)
    
    # Компактные версии сообщений (до 100 символов каждое)
    compact_msgs = []
    for msg in last_k_msgs:
        norm = normalize_entities(msg)
        compact = norm[:100] if len(norm) > 100 else norm
        
        separator = "\n---\n" if compact_msgs else ""
        
        if remaining < len(separator) + len(compact):
            break
        
        compact_msgs.append(compact)
        remaining -= len(separator) + len(compact)
    
    if compact_msgs:
        capsule = header + "\n---\n".join(compact_msgs)
    else:
        capsule = header
    
    return capsule


# WHITELIST АНАЛИЗ

def count_whitelist_hits(text: str) -> Tuple[int, int, int]:
    """
    Подсчитывает совпадения с whitelist-словарями.
    
    Используется для анти-паттернов: если сообщение содержит термины
    из легитимных категорий, снижаем подозрительность.
    
    Args:
        text: сырой текст сообщения
        
    Returns:
        (store_hits, order_hits, brand_hits) - количество уникальных совпадений
    """
    text_lower = text.lower()
    
    store_hits = sum(1 for term in _STORE_TERMS_LOWER if term in text_lower)
    order_hits = sum(1 for term in _ORDER_TERMS_LOWER if term in text_lower)
    brand_hits = sum(1 for term in _BRAND_TERMS_LOWER if term in text_lower)
    
    return store_hits, order_hits, brand_hits


# ОБРАТНАЯ СОВМЕСТИМОСТЬ (старая функция для TF-IDF/Keyword)

def prepare_for_embedding(text: str) -> str:
    """
    LEGACY: простая нормализация для одиночного сообщения.
    
    Теперь рекомендуется использовать build_context_capsule() или
    normalize_entities() + префикс "passage:" вручную.
    
    Args:
        text: исходный текст сообщения
        
    Returns:
        нормализованный текст с префиксом 'passage: '
    """
    normalized = normalize_entities(text)
    return f"passage: {normalized}"


# ПАТТЕРН-ФИЧИ (для мета-классификатора)


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
    
    РАСШИРЕНО: добавлены бинарные флаги для сущностей (has_phone, has_url, has_email)
    и whitelist-хиты.
    
    Returns:
        dict с ключами:
        - has_phone, has_url, has_email: бинарные флаги наличия сущностей
        - has_money, money_count: деньги/валюты
        - has_age: возрастные ограничения
        - has_cta_plus: CTA с плюсом/стартом
        - has_dm: призыв к ЛС
        - has_contact: любые контакты (URL/TG/PHONE/EMAIL)
        - has_remote: удалённая работа
        - has_legal: легальность/честность
        - has_casino: казино/ставки
        - obfuscation_ratio: обфускация текста
        - whitelist_hits_store, whitelist_hits_order, whitelist_hits_brand
    """
    text_lower = text.lower()
    
    # НОВОЕ: Бинарные флаги для сущностей
    has_phone = bool(re.search(
        r'(?:\+\d{1,3})?[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{2}[-.\s]?\d{2}|\+\d{10,15}',
        text
    ))
    
    has_url = bool(re.search(r'https?://|www\.', text, re.IGNORECASE))
    
    has_email = bool(re.search(
        r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
        text,
        re.IGNORECASE
    ))
    
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
    
    # Контакты (URL, TG, телефон, email) - legacy
    has_contact = has_url or has_phone or has_email or bool(
        re.search(r't\.me/|@[a-zA-Z0-9_]{3,}', text)
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
    
    # НОВОЕ: Whitelist-хиты
    whitelist_hits_store, whitelist_hits_order, whitelist_hits_brand = count_whitelist_hits(text)
    
    return {
        # Сущности (бинарные)
        'has_phone': has_phone,
        'has_url': has_url,
        'has_email': has_email,
        # Паттерны
        'has_money': has_money,
        'money_count': money_count,
        'has_age': has_age,
        'has_cta_plus': has_cta_plus,
        'has_dm': has_dm,
        'has_contact': has_contact,  # legacy
        'has_remote': has_remote,
        'has_legal': has_legal,
        'has_casino': has_casino,
        'obfuscation_ratio': obfuscation_ratio,
        # Whitelist анти-паттерны
        'whitelist_hits_store': whitelist_hits_store,
        'whitelist_hits_order': whitelist_hits_order,
        'whitelist_hits_brand': whitelist_hits_brand,
    }
