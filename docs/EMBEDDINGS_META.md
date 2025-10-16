# EMBEDDINGS & META — устройство и эксплуатация

Этот документ глубже объясняет, как в системе устроены:
— контекстные эмбеддинги (какие капсулы, кто и как их считает),
— мета‑классификатор (фичи, центроиды, прототипы, калибровка),
— Policy Engine (как используется p_spam и какие действуют понижающие множители),
— конфигурация и практические шаги по обучению/отладке/апгрейду.

Быстрый обзор общей архитектуры см. в `docs/ARCHITECTURE.md:1`.

Термины и источники
- `FilterCoordinator` — оркестратор фильтров, капсул и истории. Источник: `core/coordinator.py:1`.
- `AnalysisResult` — объединённый результат фильтров/капсул/эмбеддингов/мета. Источник: `core/types.py:1`.
- `FilterResult` — результат одного фильтра (name, score, confidence, details). Источник: `core/types.py:1`.
- `MessageMetadata` — метаданные сообщения (reply_to_staff, is_forwarded, и т.д.). Источник: `core/types.py:1`.
- `KeywordFilter`, `TfidfFilter`, `EmbeddingFilter` — реализации фильтров. Источники: `filters/keyword.py:1`, `filters/tfidf.py:1`, `filters/embedding.py:1`.
- `MetaClassifier` — расчёт `p_spam` и отладочной информации. Источник: `services/meta_classifier.py:1`.
- `PolicyEngine` — выбор действия (APPROVE/NOTIFY/DELETE/KICK). Источник: `services/policy.py:1`.

---

## Эмбеддинги: капсулы, провайдер, кэш

### Капсулы
Капсулы формируются в `utils/textprep.py:1` и используются для мультиязычных моделей (e5‑семейство и аналоги). Все тексты нормализуются с заменой сущностей на плейсхолдеры для лучшей устойчивости.

- Message (E_msg):
  - `prepare_for_embedding(text)` или вручную: `"passage: " + normalize_entities(text)`.
- Context (E_ctx):
  - `build_context_capsule(message, history, metadata, max_chars)`
  - Заголовок с флагами `[CTX:n][reply_to_staff][is_forwarded][author_is_admin][is_channel_announcement]`,
    затем текущее сообщение и недавняя история (разделитель `---`).
- User (E_user):
  - `build_user_capsule(last_k_msgs, max_chars)` — компактная сводка истории пользователя
    (каждое сообщение укорачивается до ~100 символов).

Ключевые теги нормализации: `<URL>`, `<TG>`, `<PHONE>`, `<EMAIL>`, `<MONEY>`, `<AGE>`.

Референсы:
- Капсулы/нормализация: `utils/textprep.py:1`
- Интеграция и история: `core/coordinator.py:1`

#### Откуда берутся эти функции и что они делают

- `normalize_entities(text)` — нормализует текст и заменяет сущности на стабильные плейсхолдеры. Источник: `utils/textprep.py:1`.
- `prepare_for_embedding(text)` — оборачивает нормализованный текст префиксом `passage: `, что соответствует рекомендациям e5‑моделей (для «документа»/пассажа). Источник: `utils/textprep.py:1`.
- `build_context_capsule(message, history, metadata, max_chars)` — формирует контекстную капсулу с бинарными флагами и историей чата (новые → старые), усечение по лимиту символов. Источник: `utils/textprep.py:1`.
- `build_user_capsule(last_k_msgs, max_chars)` — компактная капсула истории пользователя. Источник: `utils/textprep.py:1`.
- Дополнительно для мета‑фичей:
  - `extract_patterns_from_raw(text)` — извлекает паттерны (URL/PHONE/EMAIL/деньги/возраст/CTA/DM/казино/обфускация и пр.). Источник: `utils/textprep.py:1`.
  - `count_whitelist_hits(text)` — считает попадания в whitelist‑термины (store/order/brand). Источник: `utils/textprep.py:1`.

Почему префикс `passage:`?
- Для семейства e5 (и совместимых) различают «query:» и «passage:». Мы используем «passage:» для текстов сообщений/контента, чтобы модель формировала подходящие эмбеддинги документа. 

Пример использования (ручной режим; обычно всё делает координатор):

```python
from utils.textprep import (
    normalize_entities,
    prepare_for_embedding,
    build_context_capsule,
    build_user_capsule,
)

text = "Зарабатывай от $500 в день! Пиши в ЛС +"

# E_msg (два эквивалентных способа)
message_capsule = prepare_for_embedding(text)
# или вручную:
# message_capsule = f"passage: {normalize_entities(text)}"

# E_ctx
history = ["Это ответ?", "Предыдущее сообщение"]  # новые → старые
metadata = {
    'reply_to_staff': False,
    'is_forwarded': False,
    'author_is_admin': False,
    'is_channel_announcement': False,
}
context_capsule = build_context_capsule(
    message=text,
    history=history,
    metadata=metadata,
    max_chars=2048,
)

# E_user (если хотим учитывать историю пользователя)
user_capsule = build_user_capsule(last_k_msgs=["сообщение 1", "сообщение 2"], max_chars=512)
```

Термины и источники
- `E_msg`, `E_ctx`, `E_user` — эмбеддинги сообщения, контекста и истории пользователя; контейнер `EmbeddingVectors`. Источник: `core/types.py:1`.
- `EmbeddingFilter.compute_embeddings_multi()` — батч‑получение векторов по капсулам; возвращает `EmbeddingVectors` и debug. Источник: `filters/embedding.py:260`.
- `EmbeddingProvider` — абстракция провайдера эмбеддингов. Источник: `filters/embedding.py:100`.
- `OllamaProvider` — реализация провайдера через Ollama API. Источник: `filters/embedding.py:100`.
- `EmbeddingCache` — LRU‑кэш E_user с TTL и инвалидацией. Источник: `filters/embedding.py:1`.

### Провайдер эмбеддингов и батч
`EmbeddingFilter` (в `filters/embedding.py:1`) использует абстрактный `EmbeddingProvider` и реализацию `OllamaProvider`.

- Режимы: `ollama` | `local` | `disabled` (см. .env → `EMBEDDING_MODE`).
- Пакетный расчёт: `get_embeddings_batch(texts, timeout_ms)` параллелит HTTP‑запросы (httpx) и возвращает список векторов (или None для неудачных кейсов).
- Таймаут/дефолты: берутся из конфигурации (`EMBEDDING_TIMEOUT_MS` и др.).
- Обработка ошибок: `graceful degradation` — отсутствие E_ctx/E_user не блокирует пайплайн.

Рекомендации:
- Держать капсулы в разумных пределах (`CONTEXT_MAX_TOKENS` → `max_chars ≈ tokens * 4`).
- Следить за тайм‑аутами и размером батчей к Ollama.

### LRU‑кэш для E_user
Чтобы не пересчитывать историю пользователя на каждом сообщении, `EmbeddingFilter` хранит E_user в LRU‑кэше с TTL: `EmbeddingCache(max_size, ttl_minutes)`.

- Инвалидация: при новом сообщении пользователя кэш сбрасывается для его user_id (`invalidate_user_cache`).
- Статистика: `get_cache_stats()`.

Референсы:
- Кэш: `filters/embedding.py:1`
- Параметры из .env: `config/config.py:1`

---

## Мета‑классификатор: фичи, центроиды, прототипы, калибровка

Цель — собрать все сигналы в единую калиброванную вероятность спама `p_spam` и предоставить детальный `meta_debug` для объяснимости.

Реализация: `services/meta_classifier.py:1`.

### Фичи (в порядке `feature_spec.json`)
Примерный состав (конкретный порядок читается из `models/feature_spec.json`):
- E_msg: `sim_spam_msg`, `sim_ham_msg`, `delta_msg` (косинусы к центроидам и их разность).
- E_ctx: `sim_spam_ctx`, `sim_ham_ctx`, `delta_ctx`.
- E_user: `sim_spam_user`, `sim_ham_user`, `delta_user` (если доступен).
- Прототипы (K‑means): 7 семейств спама и 4 — легит (по E_msg).
- Скоринги старых фильтров: `kw_score`, `tfidf_score` (вспомогательно).
- Паттерны сырого текста: сущности (URL/PHONE/EMAIL), деньги (+счётчик), возраст, CTA/DM, удалёнка, «легально», казино, обфускация.
- Контекстные флаги: reply_to_staff, is_forwarded, author_is_admin, is_channel_announcement.
- Whitelist анти‑паттерны: `whitelist_hits_store/order/brand` (понижают подозрительность).

В отладке (`meta_debug`) возвращаются значения косинусов/дельт, fired‑паттерны, whitelist‑хиты и топ‑вклады фичей (если обучена LogReg):
`_compute_top_contributions` берёт top‑k по |value * coef|.

### Центроиды и прототипы
- Центроиды (spam/ham): средние вектора класса по тренировочной выборке — сохраняются в `models/centroids.npz`.
- Прототипы семейства: K‑means центры по спаму/легиту, хранятся в `models/prototypes.npz`.
- Сборка фичей и тренинг — в `scripts/train_meta_context.py:1` (контекстный вариант) или `scripts/train_meta.py:1` (базовый).

### Калибровка вероятностей
Используется `CalibratedClassifierCV` для приведения сырых баллов к корректным вероятностям:
- артефакт — `models/meta_calibrator.joblib`.
- основная модель — `models/meta_model.joblib` (LogisticRegression или иной классификатор).

### Артефакты и загрузка
`MetaClassifier._load_artifacts()` ищет и загружает артефакты из `models/` (пути регулируются .env, см. `config/config.py:1`).
Статус можно посмотреть через `/meta_info`.

Термины и источники
- `centroids.npz` — `spam_centroid`, `ham_centroid` (numpy массивы). Использование: `services/meta_classifier.py:1`; генерация: `scripts/train_meta_context.py:260`.
- `prototypes.npz` — K‑means центры семейств (spam/legit). Использование: `services/meta_classifier.py:1`; генерация: `scripts/train_meta_context.py:260`.
- `feature_spec.json` — порядок фичей для мета‑классификатора. Источник: `models/feature_spec.json` (сохраняется тренингом).
- `predict_proba(text, analysis)` — API мета‑классификатора (возврат `p_spam`, `meta_debug`). Источник: `services/meta_classifier.py:300`.
- `top_features` — топ‑вклады (value*coef) при наличии `coef_` у модели. Источник: `services/meta_classifier.py:300`.

---

## Policy Engine: режимы, пороги, понижающие множители

Реализация: `services/policy.py:1`.

Режимы работы:
- `manual`: только NOTIFY при `p_spam ≥ META_NOTIFY`, DELETE/KICK запрещены.
- `semi-auto`: NOTIFY + DELETE разрешены, KICK запрещён.
- `auto`: доступны все действия (KICK/DELETE/NOTIFY) по своим порогам.

Пороги задаются в .env: `META_NOTIFY`, `META_DELETE`, `META_KICK`.

Понижающие множители (уменьшают `p_spam` перед сравнением с порогами):
- `is_channel_announcement` → `META_DOWNWEIGHT_ANNOUNCEMENT`.
- `reply_to_staff` → `META_DOWNWEIGHT_REPLY_TO_STAFF`.
- whitelist‑хиты (`store|order|brand`) → `META_DOWNWEIGHT_WHITELIST`.

Graceful degradation (контекст): если не удалось получить E_ctx, Policy поднимает `META_NOTIFY` на +0.05, чтобы не «шуметь» без контекста.

Возвращаемая структура решения включает: исходный/скорректированный `p_spam`, применённые множители, использованные пороги, флаг деградации и текстовое объяснение.

Термины и источники
- `decide_action(analysis)` — возвращает `(Action, decision_details)`. Источник: `services/policy.py:1`.
- `Action` — перечисление действий (approve/notify/delete/kick). Источник: `core/types.py:1`.
- `downweights` — понижающие множители (announcement/reply_to_staff/whitelist). Источник: `.env` → `config/config.py:1`; логика применения: `services/policy.py:1`.
- `degraded_ctx` — признак отсутствия E_ctx, влияет на notify‑порог. Источник: `filters/embedding.py:260` → учитывается Policy.

---

## Конфигурация и окружение

Настройки хранятся в `.env` и парсятся в `config/config.py:1`. Важные параметры:

- Embeddings:
  - `EMBEDDING_MODE=ollama|local|disabled`
  - `OLLAMA_MODEL`, `OLLAMA_BASE_URL`
  - `EMBEDDING_TIMEOUT_MS`, `EMBEDDING_CACHE_TTL_MIN`
  - `EMBEDDING_ENABLE_USER=true|false`
  - Контекст: `CONTEXT_HISTORY_N`, `CONTEXT_MAX_TOKENS`

- Policy & Meta:
  - `POLICY_MODE=manual|semi-auto|auto`
  - `META_NOTIFY`, `META_DELETE`, `META_KICK`
  - `META_DOWNWEIGHT_ANNOUNCEMENT`, `META_DOWNWEIGHT_REPLY_TO_STAFF`, `META_DOWNWEIGHT_WHITELIST`
  - Пути артефактов: `CENTROIDS_PATH`, `PROTOTYPES_PATH`, `META_MODEL_PATH`, `META_CALIBRATOR_PATH`

Для runtime‑изменений порогов/множителей используется `config/runtime.py:1` + команды `/setpolicy`, `/setthreshold`, `/setdownweight`, `/resetconfig`.

Термины и источники
- `settings` — singleton с параметрами из .env. Источник: `config/config.py:1`.
- `runtime_config` — runtime‑оверрайды и трекинг изменений. Источник: `config/runtime.py:1`.
- Команды `/setpolicy`, `/setthreshold`, `/setdownweight`, `/resetconfig` — обработчики в `bot/handlers.py:380`.

---

## Обучение и проверка

1) TF‑IDF
- Датасет: `data/messages.csv` (message,label)
- Обучение/оценка: `python scripts/train_tfidf.py`

2) Мета‑классификатор (контекстный)
- Требует запущенного Ollama и настроенных `.env` (`OLLAMA_MODEL`, `OLLAMA_BASE_URL`).
- Запуск: `python scripts/train_meta_context.py`
- Артефакты: в `models/` (см. выше).
- В продакшене замените временные заглушки kw/tfidf‑скоров на реальные: интегрируйте вызов фильтров в тренинг.

Проверка готовности
- `/status` — статус фильтров/включения embedding.
- `/meta_info` — готовность мета‑классификатора, фичи, калибратор, центроиды.

Термины и источники
- `DatasetManager` — операции с `data/messages.csv` (append/size/rows). Источник: `services/dataset.py:1`.
- `train_meta_context.py` — контекстное обучение (капсулы, эмбеддинги, центроиды/прототипы, фичи, калибровка). Источник: `scripts/train_meta_context.py:1`.
- `train_meta.py` — базовый вариант обучения без контекста. Источник: `scripts/train_meta.py:1`.
- `train_tfidf.py` — обучение TF‑IDF. Источник: `scripts/train_tfidf.py:1`.

---

## Эксплуатация и отладка

- Режим карточек: упрощённая или детальная (`DETAILED_DEBUG_INFO=true` включает подробный debug c top‑features и контекстными флагами).
- История и капсулы: координатор держит историю чата/пользователя; контролируйте размер контекста (`CONTEXT_HISTORY_N`, `CONTEXT_MAX_TOKENS`).
- Кэш E_user: смотрите статистику и учитывайте TTL; инвалидация происходит при каждом новом сообщении пользователя.
- Retrain TF‑IDF: авто по `RETRAIN_THRESHOLD` или вручную `/retrain`.

Типовые проблемы
- `httpx not installed`: добавьте в зависимости или установите `pip install httpx`.
- Ollama недоступен: проверьте `OLLAMA_BASE_URL`, запущен ли сервер и модель.
- MetaClassifier не готов: обучите `train_meta_context.py` и проверьте пути артефактов.

Термины и источники
- `DETAILED_DEBUG_INFO` — включает подробные карточки по умолчанию. Источник: `.env` → `config/config.py:1`.
- `RETRAIN_THRESHOLD` — порог авто‑переобучения TF‑IDF. Источник: `.env` → `config/config.py:1`; учёт в `filters/tfidf.py:1`.