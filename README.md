# 🧠 NeuroVT

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-3.1+-green.svg)](https://flask.palletsprojects.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**NeuroVT** — это модульная система для создания AI-втубера с поддержкой голосового ввода (STT), синтеза речи (TTS), языковых моделей (LLM) и веб-интерфейса на Flask.

> 🎙️ Говорите — AI отвечает голосом.  
> 🧩 Модульная архитектура — легко добавлять новые функции.  
> 🌐 Управление через браузер.

## ✨ Возможности

- 🎤 **STT (Speech-to-Text)** — распознавание речи через `faster-whisper` с горячими клавишами
- 🔊 **TTS (Text-to-Speech)** — синтез речи через Silero с очередью и историей
- 🧠 **LLM (Large Language Model)** — подключение к OpenRouter API с управлением контекстом
- 💬 **Чат с AI** — сессии, контекст, счётчик токенов
- ⚙️ **Веб-интерфейс** — настройки, статус, история, управление модулями
- 🖥️ **CLI-менеджер** — установка, обновление, запуск (Windows/Linux)
- 📁 **HTML-шаблоны модулей** — каждый модуль может иметь свои шаблоны

---

## 🚀 Быстрый старт

### 1. Клонирование репозитория

```bash
git clone https://github.com/FreshLend/NeuroVT.git
cd NeuroVT
```

### 2. Запуск менеджера

**Windows** 
Дважды кликните `run.bat`

**Linux**:
```bash
chmod +x run.sh
./run.sh
```

## 📁 Структура проекта

```
NeuroVT/
├── main.py                 # Точка входа (Flask + модули)
├── run.bat / run.sh        # CLI-менеджер
├── requirements.txt        # Основные зависимости
├── core/
│   ├── event_bus.py        # Шина событий между модулями
│   └── module_loader.py    # Автозагрузчик модулей
├── modules/
│   ├── base_module.py      # Базовый класс для модулей
│   ├── tts_module/         # TTS (Silero)
│   │   └── templates/      # HTML-шаблоны TTS
│   ├── stt_module/         # STT (Faster-Whisper)
│   │   └── templates/      # HTML-шаблоны STT
│   └── llm_module/         # LLM (OpenRouter)
│       └── templates/      # HTML-шаблоны LLM
└── templates/
    ├── base.html           # Общий шаблон
    ├── index.html          # Главная страница (вкладки модулей)
    └── settings.html       # Страница настроек
```

## 🧩 Модули

### 🎙️ STT (Speech-to-Text)

- Модель: `faster-whisper` (small, int8)
- Горячая клавиша: `Ctrl+Shift+M` (настраивается)
- Распознаёт русский, английский и другие языки
- Выбор языка распознавания (авто/русский/английский и др.)
- Отправляет текст в LLM через событие `llm_voice_input`

### 🔊 TTS (Text-to-Speech)

- Модель: Silero (v3_1_ru)
- Голоса: Айдар, Бая, Ксения, Ксения (альт), Евгений
- Качество: 24 кГц / 48 кГц
- Очередь сообщений, история, статус воспроизведения

### 🧠 LLM (OpenRouter)

- Управление контекстом: ограничение по токенам, автообрезка истории
- Подсчёт токенов через `tiktoken`
- Fallback-ответы при отсутствии API-ключа
- Интеграция с голосовым вводом (STT)

---

## 🛠️ Установка зависимостей вручную

Если вы не используете `run.bat` / `run.sh`:

```bash
# Создать виртуальное окружение
python -m venv venv

# Активировать
source venv/bin/activate      # Linux
venv\Scripts\activate         # Windows

# Установить основные зависимости
pip install -r requirements.txt

# Установить зависимости модулей
pip install -r modules/tts_module/requirements.txt
pip install -r modules/stt_module/requirements.txt
pip install -r modules/llm_module/requirements.txt
```

## ⚙️ Настройка API-ключа OpenRouter

1. Зарегистрируйтесь на [openrouter.ai](https://openrouter.ai)
2. Создайте API-ключ
3. В веб-интерфейсе перейдите в **Настройки → LLM**
4. Вставьте ключ
5. Настройте лимит контекста (токенов) при необходимости
6. Нажмите **Сохранить все настройки**

Без ключа будут использоваться fallback-ответы (офлайн-режим).

## 🎯 Пример использования

1. **Запустите** `run.bat` или `./run.sh`
2. Выберите **Full installation (1)**
3. После установки откройте браузер с `http://localhost:5000`
4. Перейдите в **Настройки → LLM** и введите API-ключ OpenRouter
5. Вернитесь на главную и выберите вкладку **Чат с AI**
6. Нажмите **Ctrl+Shift+M**, чтобы включить микрофон
7. Скажите что-нибудь — AI ответит голосом!

---

## 🔧 Разработка модулей

### Создание своего модуля

1. Создайте папку в `modules/`, например `modules/my_module/`
2. Создайте папку `templates/` внутри неё (для HTML-шаблонов)
3. Создайте файл `my_module.py`:

```python
from modules.base_module import BaseModule
from flask import jsonify

class MyModule(BaseModule):
    name = "my_module"
    display_name = "Мой модуль"
    
    def register_routes(self):
        @self.app.route('/api/my_module/hello')
        def my_hello():
            return jsonify({"message": "Hello from my module!"})
    
    def register_main_tab(self):
        return ("Моя вкладка", self.get_template_content("main_tab.html"))
    
    def register_settings_ui(self):
        return self.get_template_content("settings.html")
    
    def on_load(self):
        print(f"[{self.display_name}] Загружен!")
        self.event_bus.subscribe("some_event", self.handle_event)
    
    def handle_event(self, data):
        print(f"Получено событие: {data}")
```

4. Добавьте `requirements.txt` в папку модуля (если нужны зависимости)
5. Создайте HTML-шаблоны в `modules/my_module/templates/`:
   - `main_tab.html` — содержимое вкладки на главной странице
   - `settings.html` — содержимое вкладки настроек
6. Перезапустите NeuroVT — модуль загрузится автоматически

### События Event Bus

| Событие | Отправитель | Данные |
|---------|-------------|--------|
| `tts_speak` | LLM / любой модуль | `{"text": "...", "source": "..."}` |
| `stt_text_ready` | STT | `{"text": "...", "is_final": true, "language": "..."}` |
| `llm_voice_input` | STT | `{"text": "...", "source": "microphone", "timestamp": "..."}` |

### Как подписаться на событие

```python
def on_load(self):
    self.event_bus.subscribe("tts_speak", self.handle_tts)

def handle_tts(self, data):
    text = data.get("text", "")
    print(f"Озвучить: {text}")
```

## 📄 Лицензия

Проект распространяется под лицензией **MIT**. Подробнее см. в файле `LICENSE`.
