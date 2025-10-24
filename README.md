# 🛡️ LifeMart Safety Bot

Telegram-бот для автоматической и полуавтоматической модерации чатов компании **LifeMart**. Использует машинное обучение для выявления спама и предоставляет модераторам удобный интерфейс для подтверждения или отклонения подозрительных сообщений.

---

## 🚀 Возможности

- 🎯 **Трехуровневая система фильтрации:**
  - 🔤 Keyword Filter - поиск по ключевым словам и паттернам
  - 📊 TF-IDF Filter - машинное обучение на основе TF-IDF векторизации
  - 🧠 Embedding Filter - семантический анализ через эмбеддинги (Mistral API или Ollama)
- 🤖 Интерактивные сообщения с кнопками "Спам" / "Бан" / "Не спам"
- 🗂️ Постоянное расширение датасета `data/messages.csv`
- 📉 Автоматическое переобучение TF-IDF модели по порогу или вручную
- ⚙️ Три режима политики: `manual`, `semi-auto`, `auto`
- 💬 Уведомления модераторов с детализированными карточками
- 🔐 Белый список доверенных пользователей
- 🌐 Поддержка облачных (Mistral API) и локальных (Ollama) эмбеддингов
- 🔄 Простая архитектура, легко расширяемая

---

## 📁 Структура проекта

```text
LifeMart_Safety/
├── bot/                     # Основная логика бота
│   ├── __init__.py
│   ├── core.py             # Ядро бота (инициализация и конфигурация)
│   ├── handlers.py         # Обработчики команд и сообщений
│   └── middleware.py       # Промежуточные слои (middleware)
│
├── config/                 # Конфигурационные файлы
│   ├── __init__.py
│   └── config.py           # Основная конфигурация проекта
│
├── data/                   # Статические данные
│   └── messages.csv        # CSV-файл с сообщениями или данными
│
├── logs/                   # Лог-файлы (папка может заполняться во время работы)
│
├── ml/                     # Модуль машинного обучения
│   ├── __init__.py
│   ├── model.pkl           # Сохранённая модель
│   ├── model.py            # Код для обучения и использования модели
│   └── registry.py         # Реестр моделей или вспомогательные функции
│
├── utils/                  # Утилиты и вспомогательные функции
│   ├── decorators.py       # Декораторы
│   └── logger.py           # Настройка логирования
│
├── .env                    # Файл окружения с переменными (в .gitignore)
├── .env.example            # Пример .env файла
├── .gitattributes
├── .gitignore              # Исключения для Git
├── LICENSE                 # Лицензия проекта
├── main.py                 # Точка входа в приложение
├── README.md               # Документация проекта
├── requirements.txt        # Список зависимостей проекта
└── train.py                # Скрипт для обучения модели
```

---

## 🧪 Установка

```bash
git clone https://github.com/your-org/LifeMart_Safety.git
cd LifeMart_Safety

python -m venv .venv
source .venv/bin/activate  # или .venv\Scripts\activate на Windows

pip install -r requirements.txt
cp .env.example .env       # отредактируйте токен и параметры
python main.py
```

---

## ⚙️ Настройки через `.env`

```dotenv
# Telegram
BOT_TOKEN=ваш_токен_бота
MODERATOR_CHAT_ID=-1002591862634
WHITELIST_USER_IDS=123456789,987654321

# Эмбеддинги (выберите один из режимов)
EMBEDDING_MODE=ollama                    # api / ollama / local / disabled
MISTRAL_API_KEY=your_key_here           # для режима api
OLLAMA_MODEL=nomic-embed-text           # для режима ollama
OLLAMA_BASE_URL=http://localhost:11434  # для режима ollama

# Политика модерации
POLICY_MODE=semi-auto                   # manual / legacy-manual / semi-auto / auto
AUTO_DELETE_THRESHOLD=0.85
AUTO_KICK_THRESHOLD=0.95
NOTIFY_THRESHOLD=0.5

# Пороги фильтров
KEYWORD_THRESHOLD=0.7
TFIDF_THRESHOLD=0.6
EMBEDDING_THRESHOLD=0.7
```

📖 **Подробнее об эмбеддингах:** см. [docs/EMBEDDING_PROVIDERS.md](docs/EMBEDDING_PROVIDERS.md)  
🔧 **Настройка Ollama:** см. [docs/OLLAMA_SETUP.md](docs/OLLAMA_SETUP.md)

---

## 👮 Режимы политики (POLICY_MODE)

- **`manual`**: все подозрительные сообщения отправляются модератору без автодействий
- **`legacy-manual`**: режим keyword -> TF-IDF без участия мета-классификатора; p_spam нового стека выводится только для сравнения.
- **`semi-auto`**: сообщения с высокой вероятностью спама удаляются автоматически, средние - модератору
- **`auto`**: полностью автоматическая модерация (удаление/бан по порогам)

---

## 🧠 Фильтры спама

### 1️⃣ Keyword Filter
Быстрая проверка по базе ключевых слов и регулярных выражений (`data/keywords.json`).

### 2️⃣ TF-IDF Filter  
Машинное обучение на основе датасета `data/messages.csv`. Использует Multinomial Naive Bayes.

### 3️⃣ Embedding Filter
Семантический анализ через векторные представления текста. Поддерживает:
- **Mistral API** - облачное решение (платное)
- **Ollama** - локальный сервер (бесплатно, приватно)
- **Disabled** - отключить семантику

Итоговый скор = взвешенная сумма всех фильтров (вес эмбеддингов 50%).

---

## 🔄 Переобучение TF-IDF модели

Сообщения, помеченные модератором как "спам", добавляются в `data/messages.csv`. После достижения `RETRAIN_THRESHOLD` новых примеров модель автоматически переобучается.

Вручную запустить переобучение (только для whitelist пользователей):

```bash
/retrain
```

Или через скрипт:

```bash
python scripts/train_tfidf.py
```

---

## 📋 Команды бота

| Команда       | Описание                                      | Доступ        |
|---------------|-----------------------------------------------|---------------|
| `/start`      | Приветственное сообщение                      | Все           |
| `/status`     | Статус фильтров и количество примеров         | Whitelist     |
| `/retrain`    | Переобучить TF-IDF модель вручную             | Whitelist     |

---

## 📦 Пример модераторской карточки

> **🚨 Подозрительное сообщение**  
>  
> 👤 **Пользователь:** Иван Иванов  
> � **Средний скор:** 0.87 | **Макс:** 0.95  
>  
> 🔤 **Keyword:** 0.85 - matched: "заработок", "деньги"  
> 📊 **TF-IDF:** 0.92 - high spam probability  
> 🧠 **Embedding:** 0.85 - Max similarity: 0.75 (заработок деньги доход...)  
>  
> 💬 **Текст:**  
> _Зарабатывай от $500 в день! Пиши в личку + и получи инструкцию!_  
>  
> 🔗 [Перейти к сообщению](https://t.me/c/...)  
>  
> [ 🚫 Спам ] [ ⛔ Бан ] [ ✅ Не спам ]

---

## 🛡️ Действия модератора

- **"🚫 Спам"**:
  - Удаляет сообщение (если не удалено)
  - Добавляет пример в `data/messages.csv` для обучения
  - Показывает уведомление в чате

- **"⛔ Бан"**:
  - Удаляет сообщение
  - Блокирует пользователя в чате
  - Добавляет в датасет
  - Публичное уведомление (если `ANNOUNCE_BLOCKS=True`)

- **"✅ Не спам"**:
  - Сообщение остаётся
  - Пример **не добавляется** в датасет
  - Карточка модератора удаляется

---


## LLM evaluation layer

- Optional final decision step powered by OpenRouter. Enable it with `LLM_EVAL_ENABLED=True` and provide `OPENROUTER_API_KEY` together with `LLM_EVAL_MODEL`.
- The service receives the results of all filters plus the serialized chat and user capsules, then returns one of the standard actions (`approve`, `notify`, `delete`, `kick`).
- Confidence threshold is controlled by `LLM_EVAL_MIN_CONFIDENCE` (default `0.55`). Below that value the policy engine sticks to meta-classifier thresholds.
- When the key is missing or the flag is `False` nothing changes; the pipeline behaves exactly as before.
- Extra knobs: `LLM_EVAL_TEMPERATURE` for sampling sharpness and `LLM_EVAL_TIMEOUT_SEC` to cap network latency.

## 🧠 Требования

- Python 3.10+
- Telegram Bot API
- Зависимости: `python-telegram-bot`, `scikit-learn`, `pandas`, `httpx`, `python-dotenv`, `nltk`
- Опционально: Ollama (для локальных эмбеддингов) или Mistral API ключ

---

## 📌 TODO

- [ ] Поддержка мультичатов с отдельными моделями
- [ ] Переход на постоянную базу вместо CSV
- [ ] Веб-панель администратора

---

## 📜 Лицензия

Проект распространяется под лицензией **MIT**.


