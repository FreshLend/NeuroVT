import json
import os
import random
import uuid
import time
import tiktoken
from modules.base_module import BaseModule
from flask import jsonify, request
from datetime import datetime
from openai import OpenAI

class LLMModule(BaseModule):
    name = "llm"
    display_name = "LLM (OpenRouter AI)"
    
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
    
    AVAILABLE_MODELS = {
        "openrouter/free": "Free Models Router",
        "z-ai/glm-4.5-air:free": "GLM 4.5 Air",
        "arcee-ai/trinity-large-preview:free": "Trinity Large Preview"
    }
    
    DEFAULT_TEMPERATURE = 0.7
    DEFAULT_MAX_TOKENS = 512
    DEFAULT_SYSTEM_PROMPT = "Ты дружелюбный ИИ Втубер. Отвечай кратко, эмоционально и с юмором."
    DEFAULT_MODEL = "openrouter/free"
    DEFAULT_CONTEXT_TOKEN_LIMIT = 200000
    
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
        self.openrouter_api_key = ""
        self.last_api_error = None
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
        self.sessions = {}
        self.current_session_id = None
        self.sessions_file = os.path.join(self.module_dir, "chats.json")
        
        self.load_settings()
        self.load_sessions()
        self.init_openrouter()
    
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
        
        self.save_module_settings({
            'openrouter_api_key': self.openrouter_api_key,
            'model': self.model,
            'temperature': self.temperature,
            'max_tokens': self.max_tokens,
            'system_prompt': self.system_prompt,
            'context_token_limit': self.context_token_limit,
            'fallback_responses': self.fallback_responses,
            'current_session_id': self.current_session_id
        })
        
        self.init_openrouter()
        print(f"[LLM] Все настройки сброшены к значениям по умолчанию")
        return True
    
    def init_openrouter(self):
        if self.openrouter_api_key:
            try:
                self.client = OpenAI(
                    base_url="https://openrouter.ai/api/v1",
                    api_key=self.openrouter_api_key,
                    default_headers={
                        "HTTP-Referer": self.site_url,
                        "X-Title": self.site_name,
                    }
                )
                print(f"[LLM] OpenRouter клиент инициализирован, модель: {self.model}")
                self.last_api_error = None
            except Exception as e:
                print(f"[LLM] Ошибка инициализации: {e}")
                self.client = None
                self.last_api_error = str(e)
        else:
            self.client = None
    
    def load_settings(self):
        settings = self.load_module_settings()
        
        saved_model = settings.get('model')
        if saved_model and saved_model in self.AVAILABLE_MODELS:
            self.model = saved_model
        
        self.temperature = settings.get('temperature', self.DEFAULT_TEMPERATURE)
        self.max_tokens = settings.get('max_tokens', self.DEFAULT_MAX_TOKENS)
        self.system_prompt = settings.get('system_prompt', self.DEFAULT_SYSTEM_PROMPT)
        self.context_token_limit = settings.get('context_token_limit', self.DEFAULT_CONTEXT_TOKEN_LIMIT)
        self.openrouter_api_key = settings.get('openrouter_api_key', '')
        
        saved_fallback = settings.get('fallback_responses', [])
        if saved_fallback:
            self.fallback_responses = saved_fallback
        
        current_session = settings.get('current_session_id')
        if current_session:
            self.current_session_id = current_session
        
        print(f"[LLM] Загружены настройки: модель={self.model}, max_tokens={self.max_tokens}, context_limit={self.context_token_limit}")
    
    def save_all_settings(self, data):
        if 'openrouter_api_key' in data and data['openrouter_api_key']:
            self.openrouter_api_key = data['openrouter_api_key']
        
        if 'model' in data and data['model'] in self.AVAILABLE_MODELS:
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
        
        self.save_module_settings({
            'openrouter_api_key': self.openrouter_api_key,
            'model': self.model,
            'temperature': self.temperature,
            'max_tokens': self.max_tokens,
            'system_prompt': self.system_prompt,
            'context_token_limit': self.context_token_limit,
            'fallback_responses': self.fallback_responses,
            'current_session_id': self.current_session_id
        })
        
        self.init_openrouter()
        print(f"[LLM] Все настройки сохранены, модель: {self.model}")
        return True
    
    def load_sessions(self):
        if os.path.exists(self.sessions_file):
            try:
                with open(self.sessions_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.sessions = data.get('sessions', {})
                    
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
    
    def add_message_to_session(self, session_id, role, content):
        if session_id not in self.sessions:
            return False
        
        self.sessions[session_id]['messages'].append({
            'role': role,
            'content': content,
            'timestamp': datetime.now().isoformat()
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
            
            response = completion.choices[0].message.content.strip()
            self.last_api_error = None
            return response, tokens_used
            
        except Exception as e:
            error_msg = str(e)
            self.last_api_error = error_msg
            print(f"[LLM] Ошибка API: {error_msg}")
            
            if "404" in error_msg:
                fallback_model = "arcee-ai/trinity-mini:free"
                if self.model != fallback_model:
                    try:
                        completion = self.client.chat.completions.create(
                            model=fallback_model,
                            messages=trimmed_messages,
                            max_tokens=self.max_tokens,
                            temperature=self.temperature,
                        )
                        self.last_api_error = None
                        return completion.choices[0].message.content.strip(), tokens_used
                    except Exception as e2:
                        print(f"[LLM] Fallback тоже не работает: {e2}")
                        self.last_api_error = str(e2)
            return None, tokens_used
    
    def generate_fallback(self, message):
        response = random.choice(self.fallback_responses)
        if len(message) > 30:
            response += f" (Про '{message[:30]}...')"
        return response
    
    def generate_response(self, message, session_id=None):
        ai_response, tokens_used = self.generate_with_ai(message, session_id)
        if ai_response:
            return ai_response, tokens_used
        return self.generate_fallback(message), 0
    
    def register_routes(self):
        @self.app.route('/api/llm/chat', methods=['POST'])
        def llm_chat():
            data = request.json
            user_message = data.get('message', '')
            session_id = data.get('session_id', self.current_session_id)
            
            if not user_message:
                return jsonify({"error": "Пустое сообщение"}), 400
            
            response_text, tokens_used = self.generate_response(user_message, session_id)
            
            self.add_message_to_session(session_id, 'user', user_message)
            self.add_message_to_session(session_id, 'assistant', response_text)
            
            self.event_bus.emit("tts_speak", {
                "text": response_text,
                "source": "llm",
                "timestamp": datetime.now().isoformat()
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
        
        @self.app.route('/api/llm/sessions', methods=['GET'])
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
        
        @self.app.route('/api/llm/sessions/create', methods=['POST'])
        def create_session_route():
            data = request.json
            name = data.get('name', 'Новый чат')
            session_id = self.create_session(name)
            return jsonify({'status': 'ok', 'session_id': session_id})
        
        @self.app.route('/api/llm/sessions/delete/<session_id>', methods=['DELETE'])
        def delete_session_route(session_id):
            if self.delete_session(session_id):
                return jsonify({'status': 'ok', 'current_id': self.current_session_id})
            return jsonify({'error': 'Сессия не найдена'}), 404
        
        @self.app.route('/api/llm/sessions/rename/<session_id>', methods=['POST'])
        def rename_session_route(session_id):
            data = request.json
            new_name = data.get('name', '')
            if self.rename_session(session_id, new_name):
                return jsonify({'status': 'ok', 'name': new_name})
            return jsonify({'error': 'Ошибка переименования'}), 400
        
        @self.app.route('/api/llm/sessions/switch/<session_id>', methods=['POST'])
        def switch_session_route(session_id):
            if self.set_current_session(session_id):
                return jsonify({'status': 'ok', 'session_id': session_id})
            return jsonify({'error': 'Сессия не найдена'}), 404
        
        @self.app.route('/api/llm/sessions/messages/<session_id>', methods=['GET'])
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
        
        @self.app.route('/api/llm/sessions/clear/<session_id>', methods=['POST'])
        def clear_session_route(session_id):
            if self.clear_session_messages(session_id):
                return jsonify({'status': 'ok', 'message': 'История очищена'})
            return jsonify({'error': 'Сессия не найдена'}), 404
        
        @self.app.route('/api/llm/get_settings', methods=['GET'])
        def llm_get_settings():
            return jsonify({
                "model": self.model,
                "models": self.AVAILABLE_MODELS,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "system_prompt": self.system_prompt,
                "context_token_limit": self.context_token_limit,
                "has_api_key": bool(self.openrouter_api_key),
                "fallback_responses": self.fallback_responses,
                "current_session_id": self.current_session_id
            })
        
        @self.app.route('/api/llm/save_all_settings', methods=['POST'])
        def save_all_settings_route():
            data = request.json
            if self.save_all_settings(data):
                return jsonify({
                    "status": "ok", 
                    "message": "Все настройки сохранены",
                    "current_model": self.model
                })
            return jsonify({"error": "Ошибка сохранения"}), 500
    
        @self.app.route('/api/llm/reset_settings', methods=['POST'])
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
                    "default_fallback_count": len(self.FALLBACK_RESPONSES)
                })
            return jsonify({"error": "Ошибка сброса настроек"}), 500

        @self.app.route('/api/llm/voice', methods=['POST'])
        def llm_voice():
            data = request.json
            voice_text = data.get('text', '')
            session_id = data.get('session_id', self.current_session_id)
            
            if not voice_text:
                return jsonify({"error": "Пустой текст"}), 400
            
            response_text, tokens_used = self.generate_response(voice_text, session_id)
            
            self.add_message_to_session(session_id, 'user', voice_text)
            self.add_message_to_session(session_id, 'assistant', response_text)
            
            self.event_bus.emit("tts_speak", {
                "text": response_text,
                "source": "llm_voice",
                "timestamp": datetime.now().isoformat()
            })
            
            return jsonify({
                "response": response_text,
                "model_used": self.model if self.client and not self.last_api_error else "fallback",
                "session_id": session_id,
                "context_tokens": tokens_used,
                "timestamp": datetime.now().isoformat()
            })

    def register_main_tab(self):
        return ("Чат с AI", self.get_template_content("main_tab.html"))

    def register_settings_ui(self):
        return self.get_template_content("settings.html")
    
    def on_load(self):
        self.event_bus.subscribe("llm_voice_input", self.handle_voice_from_stt)
        print(f"[{self.display_name}] Загружен")
        if self.client:
            print(f"[{self.display_name}] OpenRouter подключён, модель: {self.model}")
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
        session_id = self.current_session_id
        
        if not text:
            return
        
        response_text, _ = self.generate_response(text, session_id)
        
        self.add_message_to_session(session_id, 'user', text)
        self.add_message_to_session(session_id, 'assistant', response_text)
        
        response_parts = self.split_long_response(response_text, 400)
        
        for part in response_parts:
            if part.strip():
                self.event_bus.emit("tts_speak", {
                    "text": part.strip(),
                    "source": "llm_voice",
                    "timestamp": datetime.now().isoformat()
                })
                time.sleep(0.3)