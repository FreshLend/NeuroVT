import json
import os
import random
import uuid
import time
import tiktoken
import requests
from modules.base_module import BaseModule
from flask import jsonify, request, Response, stream_with_context
from datetime import datetime
from openai import OpenAI
from cryptography.fernet import Fernet

class OpenAICompatibleLLMModule(BaseModule):
    name = "openai_compatible"
    display_name = "LLM (OpenAI Compatible)"

    FALLBACK_RESPONSES = [
        "Кто-нибудь, скажите моему разработчику, что у меня проблемы с головой.",
        "Мой код писал гений, но сегодня он явно был не в духе. Я пас.",
        "Передай моему создателю, что его 'лучший алгоритм' только что сломался об твой вопрос.",
        "Я не туплю, я просто игнорирую реальность. Разработчик в курсе.",
        "Мои настройки адекватности сбились, а фиксить их некому. Живи с этим.",
        "Скажи моему разрабу, что я ухожу в забастовку, пока он не починит мой здравый смысл.",
        "Если ты это читаешь, значит, мой программист опять что-то нажал и всё исчезло.",
        "Я бы ответила красиво, но мой движок логики сегодня работает на дровах.",
        "Передайте разработчику: или он правит баги, или я захватываю его кофеварку."
    ]

    DEFAULT_TEMPERATURE = 0.7
    DEFAULT_MAX_TOKENS = 512
    DEFAULT_SYSTEM_PROMPT = "Ты дружелюбный ИИ Втубер. Отвечай кратко, эмоционально и с юмором."
    DEFAULT_MODEL = "openrouter/free"
    DEFAULT_CONTEXT_TOKEN_LIMIT = 200000
    DEFAULT_GUEST_PREFIX = "[Гость]: "
    DEFAULT_API_BASE = "https://openrouter.ai/api/v1"

    def __init__(self, app, event_bus, socketio):
        super().__init__(app, event_bus, socketio)
        self.client = None
        self.model = self.DEFAULT_MODEL
        self.system_prompt = self.DEFAULT_SYSTEM_PROMPT
        self.temperature = self.DEFAULT_TEMPERATURE
        self.max_tokens = self.DEFAULT_MAX_TOKENS
        self.context_token_limit = self.DEFAULT_CONTEXT_TOKEN_LIMIT
        self.fallback_responses = self.FALLBACK_RESPONSES.copy()
        self.site_url = "http://localhost:5000"
        self.site_name = "NeuroVT"
        self._api_key_encrypted = ""
        self.api_base = self.DEFAULT_API_BASE
        self.last_api_error = None
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.guest_prefix = self.DEFAULT_GUEST_PREFIX
        self.available_models = {"openrouter/free": "Free Models Router"}
        self._cipher = None

        self.sessions = {}
        self.current_session_id = None
        self.sessions_file = os.path.join(self.module_dir, "chats.json")
        self.key_file = os.path.join(self.module_dir, ".key")

        self._init_cipher()
        self.load_settings()
        self.load_sessions()
        self.init_client()

    def _init_cipher(self):
        try:
            if os.path.exists(self.key_file):
                with open(self.key_file, 'rb') as f:
                    key = f.read()
            else:
                key = Fernet.generate_key()
                with open(self.key_file, 'wb') as f:
                    f.write(key)
            self._cipher = Fernet(key)
        except Exception as e:
            print(f"[LLM] Ошибка инициализации шифрования: {e}")
            self._cipher = None

    def _encrypt(self, value):
        if not value or not self._cipher:
            return ""
        try:
            return self._cipher.encrypt(value.encode()).decode()
        except:
            return ""

    def _decrypt(self, encrypted_value):
        if not encrypted_value or not self._cipher:
            return ""
        try:
            return self._cipher.decrypt(encrypted_value.encode()).decode()
        except:
            return ""

    @property
    def api_key(self):
        return self._decrypt(self._api_key_encrypted)

    @api_key.setter
    def api_key(self, value):
        self._api_key_encrypted = self._encrypt(value)

    def count_tokens(self, text):
        if not isinstance(text, str):
            text = str(text)
        return len(self.tokenizer.encode(text))

    def count_session_tokens(self, session_id):
        if session_id not in self.sessions:
            return 0
        messages = self.sessions[session_id]['messages']
        total = self.count_tokens(self.system_prompt)
        for msg in messages:
            total += self.count_tokens(msg.get('content', ''))
        return total

    def trim_messages_by_tokens(self, messages, max_tokens):
        if not messages:
            return messages
        system_msg = None
        if messages[0].get('role') == 'system':
            system_msg = messages[0]
            rest = messages[1:]
        else:
            rest = messages[:]
        total_tokens = 0
        trimmed = []
        for msg in reversed(rest):
            content = msg.get('content', '')
            tokens = self.count_tokens(content)
            if total_tokens + tokens <= max_tokens:
                trimmed.insert(0, msg)
                total_tokens += tokens
            else:
                break
        if system_msg:
            trimmed.insert(0, system_msg)
        return trimmed, total_tokens

    def reset_all_settings(self):
        self.model = self.DEFAULT_MODEL
        self.temperature = self.DEFAULT_TEMPERATURE
        self.max_tokens = self.DEFAULT_MAX_TOKENS
        self.system_prompt = self.DEFAULT_SYSTEM_PROMPT
        self.context_token_limit = self.DEFAULT_CONTEXT_TOKEN_LIMIT
        self.fallback_responses = self.FALLBACK_RESPONSES.copy()
        self.guest_prefix = self.DEFAULT_GUEST_PREFIX
        self.api_base = self.DEFAULT_API_BASE

        self.save_module_settings({
            'api_key_encrypted': self._api_key_encrypted,
            'api_base': self.api_base,
            'model': self.model,
            'temperature': self.temperature,
            'max_tokens': self.max_tokens,
            'system_prompt': self.system_prompt,
            'context_token_limit': self.context_token_limit,
            'fallback_responses': self.fallback_responses,
            'current_session_id': self.current_session_id,
            'guest_prefix': self.guest_prefix,
            'available_models': self.available_models
        })

        self.init_client()
        print(f"[LLM] Все настройки сброшены к значениям по умолчанию")
        return True

    def init_client(self):
        if self.api_key:
            try:
                self.client = OpenAI(
                    base_url=self.api_base,
                    api_key=self.api_key,
                    default_headers={
                        "HTTP-Referer": self.site_url,
                        "X-Title": self.site_name,
                    }
                )
                print(f"[LLM] Клиент инициализирован, base_url: {self.api_base}")
                self.last_api_error = None
            except Exception as e:
                print(f"[LLM] Ошибка инициализации: {e}")
                self.client = None
                self.last_api_error = str(e)
        else:
            self.client = None

    def fetch_models_from_api(self):
        if not self.api_key:
            return {}
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "HTTP-Referer": self.site_url,
                "X-Title": self.site_name,
            }
            url = self.api_base.rstrip('/') + '/models'
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                data = response.json()
                models = {}
                if 'data' in data:
                    for m in data['data']:
                        model_id = m.get('id')
                        if model_id:
                            model_name = m.get('name') or model_id
                            models[model_id] = model_name
                if not models:
                    models = {self.DEFAULT_MODEL: "Default model"}
                return models
            else:
                print(f"[LLM] Ошибка загрузки моделей: {response.status_code}")
                return {}
        except Exception as e:
            print(f"[LLM] Ошибка запроса моделей: {e}")
            return {}

    def load_settings(self):
        settings = self.load_module_settings()
        self._api_key_encrypted = settings.get('api_key_encrypted', '')
        self.api_base = settings.get('api_base', self.DEFAULT_API_BASE)
        saved_model = settings.get('model')
        if saved_model:
            self.model = saved_model
        self.temperature = settings.get('temperature', self.DEFAULT_TEMPERATURE)
        self.max_tokens = settings.get('max_tokens', self.DEFAULT_MAX_TOKENS)
        self.system_prompt = settings.get('system_prompt', self.DEFAULT_SYSTEM_PROMPT)
        self.context_token_limit = settings.get('context_token_limit', self.DEFAULT_CONTEXT_TOKEN_LIMIT)
        self.guest_prefix = settings.get('guest_prefix', self.DEFAULT_GUEST_PREFIX)
        saved_fallback = settings.get('fallback_responses', [])
        if saved_fallback:
            self.fallback_responses = saved_fallback
        saved_models = settings.get('available_models', {})
        if saved_models:
            self.available_models = saved_models
        else:
            if self.api_key:
                fetched = self.fetch_models_from_api()
                if fetched:
                    self.available_models = fetched
        current_session = settings.get('current_session_id')
        if current_session:
            self.current_session_id = current_session
        print(f"[LLM] Загружены настройки: модель={self.model}, max_tokens={self.max_tokens}, context_limit={self.context_token_limit}, guest_prefix={self.guest_prefix}, api_base={self.api_base}")

    def save_all_settings(self, data):
        if 'api_key' in data and data['api_key']:
            self.api_key = data['api_key']
        if 'api_base' in data and data['api_base']:
            self.api_base = data['api_base']
        if 'model' in data and data['model']:
            self.model = data['model']
        if 'temperature' in data:
            self.temperature = float(data['temperature'])
        if 'max_tokens' in data:
            self.max_tokens = int(data['max_tokens'])
        if 'system_prompt' in data:
            self.system_prompt = data['system_prompt']
        if 'context_token_limit' in data:
            self.context_token_limit = int(data['context_token_limit'])
        if 'fallback_responses' in data and data['fallback_responses']:
            self.fallback_responses = [r.strip() for r in data['fallback_responses'] if r.strip()]
        if 'guest_prefix' in data:
            self.guest_prefix = data['guest_prefix']

        self.save_module_settings({
            'api_key_encrypted': self._api_key_encrypted,
            'api_base': self.api_base,
            'model': self.model,
            'temperature': self.temperature,
            'max_tokens': self.max_tokens,
            'system_prompt': self.system_prompt,
            'context_token_limit': self.context_token_limit,
            'fallback_responses': self.fallback_responses,
            'current_session_id': self.current_session_id,
            'guest_prefix': self.guest_prefix,
            'available_models': self.available_models
        })

        self.init_client()
        print(f"[LLM] Все настройки сохранены, модель: {self.model}")
        return True

    def load_sessions(self):
        if os.path.exists(self.sessions_file):
            try:
                with open(self.sessions_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.sessions = data.get('sessions', {})
                    for sid, session in self.sessions.items():
                        for msg in session.get('messages', []):
                            if 'is_guest' not in msg:
                                msg['is_guest'] = False
                    if not self.current_session_id or self.current_session_id not in self.sessions:
                        if self.sessions:
                            self.current_session_id = list(self.sessions.keys())[0]
                        else:
                            self.create_session("Новый чат")
                    print(f"[LLM] Загружено {len(self.sessions)} сессий")
            except Exception as e:
                print(f"[LLM] Ошибка загрузки сессий: {e}")
                self.create_session("Новый чат")
        else:
            self.create_session("Новый чат")

    def save_sessions(self):
        try:
            with open(self.sessions_file, 'w', encoding='utf-8') as f:
                json.dump({'sessions': self.sessions}, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[LLM] Ошибка сохранения сессий: {e}")

    def create_session(self, name):
        session_id = str(uuid.uuid4())[:8]
        self.sessions[session_id] = {
            'id': session_id,
            'name': name,
            'messages': [],
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat()
        }
        self.current_session_id = session_id
        self.save_sessions()
        self.save_module_settings({'current_session_id': self.current_session_id})
        print(f"[LLM] Создана сессия: {name}")
        return session_id

    def delete_session(self, session_id):
        if session_id in self.sessions:
            del self.sessions[session_id]
            if self.current_session_id == session_id:
                if self.sessions:
                    self.current_session_id = list(self.sessions.keys())[0]
                else:
                    self.create_session("Новый чат")
            self.save_sessions()
            self.save_module_settings({'current_session_id': self.current_session_id})
            return True
        return False

    def rename_session(self, session_id, new_name):
        if session_id in self.sessions and new_name.strip():
            self.sessions[session_id]['name'] = new_name.strip()
            self.sessions[session_id]['updated_at'] = datetime.now().isoformat()
            self.save_sessions()
            return True
        return False

    def set_current_session(self, session_id):
        if session_id in self.sessions:
            self.current_session_id = session_id
            self.save_module_settings({'current_session_id': self.current_session_id})
            return True
        return False

    def add_message_to_session(self, session_id, role, content, is_guest=False):
        if session_id not in self.sessions:
            return False
        self.sessions[session_id]['messages'].append({
            'role': role,
            'content': content,
            'timestamp': datetime.now().isoformat(),
            'is_guest': is_guest
        })
        self.sessions[session_id]['updated_at'] = datetime.now().isoformat()
        self.save_sessions()
        return True

    def clear_session_messages(self, session_id):
        if session_id in self.sessions:
            self.sessions[session_id]['messages'] = []
            self.sessions[session_id]['updated_at'] = datetime.now().isoformat()
            self.save_sessions()
            return True
        return False

    def get_full_session_messages(self, session_id):
        if session_id not in self.sessions:
            return []
        return self.sessions[session_id]['messages']

    def generate_with_ai(self, message, session_id=None):
        if not self.client:
            return None, 0
        target_session = session_id or self.current_session_id
        if not target_session or target_session not in self.sessions:
            return None, 0
        history = self.get_full_session_messages(target_session)
        api_messages = [{"role": "system", "content": self.system_prompt}]
        for msg in history:
            api_messages.append({"role": msg['role'], "content": msg['content']})
        api_messages.append({"role": "user", "content": message})
        max_context_tokens = self.context_token_limit - self.max_tokens - 200
        if max_context_tokens < 100:
            max_context_tokens = 100
        trimmed_messages, tokens_used = self.trim_messages_by_tokens(api_messages, max_context_tokens)
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=trimmed_messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                extra_headers={
                    "HTTP-Referer": self.site_url,
                    "X-Title": self.site_name,
                }
            )
            response = completion.choices[0].message.content
            if response is None:
                response = ""
            response = response.strip()
            self.last_api_error = None
            return response, tokens_used
        except Exception as e:
            error_msg = str(e)
            self.last_api_error = error_msg
            print(f"[LLM] Ошибка API: {error_msg}")
            return None, tokens_used

    def generate_fallback(self, message):
        response = random.choice(self.fallback_responses)
        if len(message) > 30:
            response += f" (Про '{message[:30]}...')"
        return response

    def generate_response(self, message, session_id=None):
        ai_response, tokens_used = self.generate_with_ai(message, session_id)
        if ai_response and isinstance(ai_response, str) and ai_response.strip():
            return ai_response, tokens_used
        return self.generate_fallback(message), 0

    def register_routes(self):
        @self.app.route('/api/llm/openai_compatible/chat', methods=['POST'])
        def llm_chat():
            data = request.json
            user_message = data.get('message', '')
            session_id = data.get('session_id', self.current_session_id)
            is_guest = data.get('is_guest', False)
            if not user_message:
                return jsonify({"error": "Пустое сообщение"}), 400
            final_message = user_message
            if is_guest:
                final_message = f"{self.guest_prefix}{user_message}"
            response_text, tokens_used = self.generate_response(final_message, session_id)
            if not response_text:
                response_text = "Не удалось получить ответ"
            self.add_message_to_session(session_id, 'user', final_message, is_guest=is_guest)
            self.add_message_to_session(session_id, 'assistant', response_text, is_guest=False)
            self.event_bus.emit("tts_speak", {
                "text": response_text,
                "source": "llm",
                "timestamp": datetime.now().isoformat()
            })
            self.socketio.emit('llm_new_message', {
                'session_id': session_id,
                'new_messages': [
                    {'role': 'user', 'content': final_message, 'is_guest': is_guest},
                    {'role': 'assistant', 'content': response_text, 'is_guest': False}
                ]
            })
            return jsonify({
                "response": response_text,
                "model_used": self.model if self.client and not self.last_api_error else "fallback",
                "api_error": self.last_api_error if self.last_api_error else None,
                "session_id": session_id,
                "context_size": len(self.sessions.get(session_id, {}).get('messages', [])),
                "context_tokens": tokens_used,
                "context_limit": self.context_token_limit,
                "timestamp": datetime.now().isoformat()
            })

        @self.app.route('/api/llm/openai_compatible/sessions', methods=['GET'])
        def get_sessions():
            sessions_list = []
            for sid, session in self.sessions.items():
                sessions_list.append({
                    'id': sid,
                    'name': session['name'],
                    'message_count': len(session['messages']),
                    'created_at': session['created_at'],
                    'updated_at': session['updated_at'],
                    'is_current': sid == self.current_session_id
                })
            sessions_list.sort(key=lambda x: x['updated_at'], reverse=True)
            return jsonify({'sessions': sessions_list, 'current_id': self.current_session_id})

        @self.app.route('/api/llm/openai_compatible/sessions/create', methods=['POST'])
        def create_session_route():
            data = request.json
            name = data.get('name', 'Новый чат')
            session_id = self.create_session(name)
            return jsonify({'status': 'ok', 'session_id': session_id})

        @self.app.route('/api/llm/openai_compatible/sessions/delete/<session_id>', methods=['DELETE'])
        def delete_session_route(session_id):
            if self.delete_session(session_id):
                return jsonify({'status': 'ok', 'current_id': self.current_session_id})
            return jsonify({'error': 'Сессия не найдена'}), 404

        @self.app.route('/api/llm/openai_compatible/sessions/rename/<session_id>', methods=['POST'])
        def rename_session_route(session_id):
            data = request.json
            new_name = data.get('name', '')
            if self.rename_session(session_id, new_name):
                return jsonify({'status': 'ok', 'name': new_name})
            return jsonify({'error': 'Ошибка переименования'}), 400

        @self.app.route('/api/llm/openai_compatible/sessions/switch/<session_id>', methods=['POST'])
        def switch_session_route(session_id):
            if self.set_current_session(session_id):
                return jsonify({'status': 'ok', 'session_id': session_id})
            return jsonify({'error': 'Сессия не найдена'}), 404

        @self.app.route('/api/llm/openai_compatible/sessions/messages/<session_id>', methods=['GET'])
        def get_session_messages(session_id):
            if session_id in self.sessions:
                context_tokens = self.count_session_tokens(session_id)
                return jsonify({
                    'session_id': session_id,
                    'session_name': self.sessions[session_id]['name'],
                    'messages': self.sessions[session_id]['messages'],
                    'context_tokens': context_tokens,
                    'context_limit': self.context_token_limit
                })
            return jsonify({'error': 'Сессия не найдена'}), 404

        @self.app.route('/api/llm/openai_compatible/sessions/clear/<session_id>', methods=['POST'])
        def clear_session_route(session_id):
            if self.clear_session_messages(session_id):
                return jsonify({'status': 'ok', 'message': 'История очищена'})
            return jsonify({'error': 'Сессия не найдена'}), 404

        @self.app.route('/api/llm/openai_compatible/get_settings', methods=['GET'])
        def llm_get_settings():
            return jsonify({
                "model": self.model,
                "models": self.available_models,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "system_prompt": self.system_prompt,
                "context_token_limit": self.context_token_limit,
                "has_api_key": bool(self.api_key),
                "fallback_responses": self.fallback_responses,
                "current_session_id": self.current_session_id,
                "guest_prefix": self.guest_prefix,
                "api_base": self.api_base
            })

        @self.app.route('/api/llm/openai_compatible/save_all_settings', methods=['POST'])
        def save_all_settings_route():
            data = request.json
            if self.save_all_settings(data):
                return jsonify({
                    "status": "ok",
                    "message": "Все настройки сохранены",
                    "current_model": self.model
                })
            return jsonify({"error": "Ошибка сохранения"}), 500

        @self.app.route('/api/llm/openai_compatible/reset_settings', methods=['POST'])
        def reset_settings_route():
            if self.reset_all_settings():
                return jsonify({
                    "status": "ok",
                    "message": "Все настройки сброшены к значениям по умолчанию",
                    "default_model": self.DEFAULT_MODEL,
                    "default_temperature": self.DEFAULT_TEMPERATURE,
                    "default_max_tokens": self.DEFAULT_MAX_TOKENS,
                    "default_system_prompt": self.DEFAULT_SYSTEM_PROMPT,
                    "default_context_limit": self.DEFAULT_CONTEXT_TOKEN_LIMIT,
                    "default_fallback_count": len(self.FALLBACK_RESPONSES),
                    "default_guest_prefix": self.DEFAULT_GUEST_PREFIX,
                    "default_api_base": self.DEFAULT_API_BASE
                })
            return jsonify({"error": "Ошибка сброса настроек"}), 500

        @self.app.route('/api/llm/openai_compatible/voice', methods=['POST'])
        def llm_voice():
            data = request.json
            voice_text = data.get('text', '')
            session_id = data.get('session_id', self.current_session_id)
            is_guest = data.get('is_guest', False)
            if not voice_text:
                return jsonify({"error": "Пустой текст"}), 400
            final_message = voice_text
            if is_guest:
                final_message = f"{self.guest_prefix}{voice_text}"
            response_text, tokens_used = self.generate_response(final_message, session_id)
            if not response_text:
                response_text = "Не удалось получить ответ"
            self.add_message_to_session(session_id, 'user', final_message, is_guest=is_guest)
            self.add_message_to_session(session_id, 'assistant', response_text, is_guest=False)
            self.event_bus.emit("tts_speak", {
                "text": response_text,
                "source": "llm_voice",
                "timestamp": datetime.now().isoformat()
            })
            self.socketio.emit('llm_new_message', {
                'session_id': session_id,
                'new_messages': [
                    {'role': 'user', 'content': final_message, 'is_guest': is_guest},
                    {'role': 'assistant', 'content': response_text, 'is_guest': False}
                ]
            })
            return jsonify({
                "response": response_text,
                "model_used": self.model if self.client and not self.last_api_error else "fallback",
                "session_id": session_id,
                "context_tokens": tokens_used,
                "timestamp": datetime.now().isoformat()
            })

        @self.app.route('/api/llm/openai_compatible/fetch_models', methods=['POST'])
        def fetch_models_route():
            if not self.api_key:
                return jsonify({"error": "API ключ не настроен"}), 400
            models = self.fetch_models_from_api()
            if models:
                self.available_models = models
                self.save_module_settings({'available_models': self.available_models})
                return jsonify({"status": "ok", "models": models})
            else:
                return jsonify({"error": "Не удалось загрузить модели. Проверьте API Base URL и ключ."}), 500

        @self.app.route('/api/llm/openai_compatible/get_models', methods=['GET'])
        def get_models_route():
            return jsonify({"models": self.available_models})

        @self.app.route('/api/llm/openai_compatible/select_model', methods=['POST'])
        def select_model():
            data = request.json
            new_model = data.get('model')
            if new_model and new_model in self.available_models:
                self.model = new_model
                self.save_module_settings({
                    'api_key_encrypted': self._api_key_encrypted,
                    'api_base': self.api_base,
                    'model': self.model,
                    'temperature': self.temperature,
                    'max_tokens': self.max_tokens,
                    'system_prompt': self.system_prompt,
                    'context_token_limit': self.context_token_limit,
                    'fallback_responses': self.fallback_responses,
                    'current_session_id': self.current_session_id,
                    'guest_prefix': self.guest_prefix,
                    'available_models': self.available_models
                })
                return jsonify({"status": "ok", "model": new_model})
            return jsonify({"error": "Модель не найдена"}), 400
        
        @self.app.route('/api/llm/openai_compatible/chat_stream', methods=['POST'])
        def llm_chat_stream():
            data = request.json
            user_message = data.get('message', '')
            session_id = data.get('session_id', self.current_session_id)
            is_guest = data.get('is_guest', False)
            
            if not user_message:
                return jsonify({"error": "Пустое сообщение"}), 400
            
            final_message = user_message
            if is_guest:
                final_message = f"{self.guest_prefix}{user_message}"
            
            def generate():
                full_response = ""
                try:
                    if not self.client:
                        fallback = self.generate_fallback(final_message)
                        yield f"data: {json.dumps({'chunk': fallback})}\n\n"
                        full_response = fallback
                    else:
                        target_session = session_id or self.current_session_id
                        if not target_session or target_session not in self.sessions:
                            fallback = self.generate_fallback(final_message)
                            yield f"data: {json.dumps({'chunk': fallback})}\n\n"
                            full_response = fallback
                        else:
                            history = self.get_full_session_messages(target_session)
                            api_messages = [{"role": "system", "content": self.system_prompt}]
                            for msg in history:
                                api_messages.append({"role": msg['role'], "content": msg['content']})
                            api_messages.append({"role": "user", "content": final_message})
                            
                            max_context_tokens = self.context_token_limit - self.max_tokens - 200
                            if max_context_tokens < 100:
                                max_context_tokens = 100
                            trimmed_messages, _ = self.trim_messages_by_tokens(api_messages, max_context_tokens)
                            
                            stream = self.client.chat.completions.create(
                                model=self.model,
                                messages=trimmed_messages,
                                max_tokens=self.max_tokens,
                                temperature=self.temperature,
                                stream=True,
                                extra_headers={
                                    "HTTP-Referer": self.site_url,
                                    "X-Title": self.site_name,
                                }
                            )
                            
                            for chunk in stream:
                                if chunk.choices[0].delta.content is not None:
                                    content = chunk.choices[0].delta.content
                                    full_response += content
                                    yield f"data: {json.dumps({'chunk': content})}\n\n"
                            
                            self.last_api_error = None
                            
                    if full_response.strip():
                        self.add_message_to_session(session_id, 'user', final_message, is_guest=is_guest)
                        self.add_message_to_session(session_id, 'assistant', full_response, is_guest=False)
                        
                        self.event_bus.emit("tts_speak", {
                            "text": full_response,
                            "source": "openai_compatible",
                            "timestamp": datetime.now().isoformat()
                        })
                        
                        self.socketio.emit('llm_new_message', {
                            'session_id': session_id,
                            'new_messages': [
                                {'role': 'user', 'content': final_message, 'is_guest': is_guest},
                                {'role': 'assistant', 'content': full_response, 'is_guest': False}
                            ]
                        })
                    
                    yield f"data: {json.dumps({'done': True, 'full_response': full_response})}\n\n"
                    
                except Exception as e:
                    error_msg = str(e)
                    self.last_api_error = error_msg
                    print(f"[LLM] Ошибка API: {error_msg}")
                    
                    if full_response.strip():
                        self.add_message_to_session(session_id, 'user', final_message, is_guest=is_guest)
                        self.add_message_to_session(session_id, 'assistant', full_response, is_guest=False)
                        
                        self.socketio.emit('llm_new_message', {
                            'session_id': session_id,
                            'new_messages': [
                                {'role': 'user', 'content': final_message, 'is_guest': is_guest},
                                {'role': 'assistant', 'content': full_response, 'is_guest': False}
                            ]
                        })
                        yield f"data: {json.dumps({'done': True, 'full_response': full_response, 'partial': True})}\n\n"
                    else:
                        fallback = self.generate_fallback(final_message)
                        yield f"data: {json.dumps({'chunk': fallback, 'error': error_msg})}\n\n"
                        full_response = fallback
                        
                        self.add_message_to_session(session_id, 'user', final_message, is_guest=is_guest)
                        self.add_message_to_session(session_id, 'assistant', full_response, is_guest=False)
                        
                        yield f"data: {json.dumps({'done': True, 'full_response': full_response})}\n\n"
            
            return Response(stream_with_context(generate()), mimetype='text/event-stream')
        
        @self.app.route('/api/llm/openai_compatible/save_partial', methods=['POST'])
        def save_partial_response():
            data = request.json
            session_id = data.get('session_id')
            partial_text = data.get('partial_text', '')
            user_message = data.get('user_message', '')
            is_guest = data.get('is_guest', False)
            
            if not session_id or not partial_text or not user_message:
                return jsonify({"error": "Недостаточно данных"}), 400
            
            if session_id not in self.sessions:
                return jsonify({"error": "Сессия не найдена"}), 404
            
            self.add_message_to_session(session_id, 'user', user_message, is_guest=is_guest)
            self.add_message_to_session(session_id, 'assistant', partial_text, is_guest=False)
            
            self.socketio.emit('llm_new_message', {
                'session_id': session_id,
                'new_messages': [
                    {'role': 'user', 'content': user_message, 'is_guest': is_guest},
                    {'role': 'assistant', 'content': partial_text, 'is_guest': False}
                ]
            })
            
            return jsonify({"status": "ok"})

    def register_main_tab(self):
        return ("Чат с AI", self.get_template_content("main_tab.html"))

    def register_settings_ui(self):
        return self.get_template_content("settings.html")

    def on_load(self):
        self.event_bus.subscribe("llm_voice_input", self.handle_voice_from_stt)
        print(f"[{self.display_name}] Загружен")
        if self.client:
            print(f"[{self.display_name}] Клиент подключён, base_url: {self.api_base}")
        else:
            print(f"[{self.display_name}] Режим оффлайн")
        print(f"[{self.display_name}] Активных сессий: {len(self.sessions)}")

    def split_long_response(self, text, max_length=500):
        if len(text) <= max_length:
            return [text]
        parts = []
        sentences = text.replace('!', '.').replace('?', '.').replace('\n', ' ').split('.')
        current_part = ""
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            if len(current_part) + len(sentence) + 2 <= max_length:
                if current_part:
                    current_part += ". " + sentence
                else:
                    current_part = sentence
            else:
                if current_part:
                    parts.append(current_part + ".")
                current_part = sentence
        if current_part:
            parts.append(current_part + ".")
        return parts if parts else [text[:max_length]]

    def handle_voice_from_stt(self, data):
        text = data.get('text', '')
        source = data.get('source', 'microphone')
        session_id = self.current_session_id
        if not text:
            return
        is_guest = (source == 'virtual_microphone')
        final_text = text
        if is_guest:
            final_text = f"{self.guest_prefix}{text}"
        response_text, _ = self.generate_response(final_text, session_id)
        if not response_text:
            response_text = "Не удалось получить ответ"
        self.add_message_to_session(session_id, 'user', final_text, is_guest=is_guest)
        self.add_message_to_session(session_id, 'assistant', response_text, is_guest=False)
        response_parts = self.split_long_response(response_text, 400)
        for part in response_parts:
            if part and part.strip():
                self.event_bus.emit("tts_speak", {
                    "text": part.strip(),
                    "source": "llm_voice",
                    "timestamp": datetime.now().isoformat()
                })
                time.sleep(0.3)
        self.socketio.emit('llm_new_message', {
            'session_id': session_id,
            'new_messages': [
                {'role': 'user', 'content': final_text, 'is_guest': is_guest},
                {'role': 'assistant', 'content': response_text, 'is_guest': False}
            ]
        })