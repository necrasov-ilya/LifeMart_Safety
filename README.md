# 🛡️ LifeMart Safety Bot

Telegram-бот для автоматической и полуавтоматической модерации чатов компании **LifeMart**. Использует машинное обучение для выявления спама и предоставляет модераторам удобный интерфейс для подтверждения или отклонения подозрительных сообщений.

---

## 🚀 Возможности

- 📌 Автоматическая классификация сообщений (спам / не спам)
- 🤖 Интерактивные сообщения с кнопками "Спам" / "Не спам"
- 🗂️ Постоянное расширение датасета `ml/dataset.csv`
- 📉 Переобучение модели по порогу `RETRAIN_THRESHOLD` или вручную
- ⚙️ Конфигурируемая политика обработки: `notify`, `delete`, `kick`
- 💬 Уведомления о заблокированных пользователях и действиях модераторов
- 🔐 Белый список доверенных пользователей (`WHITELIST_USER_IDS`)
- 🔄 Простая архитектура, масштабируемая на мультичаты и мультиботов

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
BOT_TOKEN=ваш_токен_бота
MODERATOR_CHAT_ID=          # ID модераторского чата
WHITELIST_USER_IDS=         # Admins
SPAM_POLICY=notify          # notify | delete | kick
RETRAIN_THRESHOLD=10        # Порог для переобучения модели
```

---

## 👮 Политика SPAM_POLICY

- `notify`: подозрительное сообщение остаётся в чате, решение — за модератором
- `delete`: сообщение удаляется сразу, но карточка отправляется в мод-чат
- `kick`: сообщение удаляется, пользователь исключается на 60 сек

---

## 🔄 Переобучение модели

Сообщения, помеченные как "спам", добавляются в `ml/dataset.csv`. После достижения количества `RETRAIN_THRESHOLD` или вручного вызова `/retrain`, модель будет переобучена.

```bash
/retrain  # вручную
```

---

## 📋 Команды

| Команда       | Описание                                 |
|---------------|------------------------------------------|
| `/status`     | Показать количество обучающих примеров   |
| `/retrain`    | Запустить переобучение модели вручную    |

---

## 📦 Пример интерактивной карточки

> **Подозрительное сообщение**  
> 👤 Иван Иванов  
> 🔗 [Перейти к сообщению](https://t.me/c/...)  
>  
> _Зарабатывай от $500 в день! Пиши +_  
>  
> 🚫 Спам | ✅ Не спам

---

## 🛡️ Поведение при решении модератора

- **"Спам"**:
  - Удаляет сообщение (если ещё не удалено)
  - Показывает в оригинальном чате:  
    > Пользователь Иван Иванов заблокирован по решению модератора.

  - Добавляет пример в датасет

- **"Не спам"**:
  - Сообщение остаётся
  - Пример в датасет **не добавляется**

---

## 🧠 Требования

- Python 3.10+
- Telegram Bot API
- `scikit-learn`, `pandas`, `python-telegram-bot`

---

## 📌 TODO

- [ ] Поддержка мультичатов с отдельными моделями
- [ ] Переход на постоянную базу вместо CSV
- [ ] Веб-панель администратора

---

## 📜 Лицензия

Проект распространяется под лицензией **MIT**.


