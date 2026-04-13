import json
import os
import random
import uuid
import time
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
    
    def __init__(self, app, event_bus):
        super().__init__(app, event_bus)
        self.client = None
        self.model = self.DEFAULT_MODEL
        self.system_prompt = self.DEFAULT_SYSTEM_PROMPT
        self.temperature = self.DEFAULT_TEMPERATURE
        self.max_tokens = self.DEFAULT_MAX_TOKENS
        self.fallback_responses = self.FALLBACK_RESPONSES.copy()
        self.site_url = "http://localhost:5000"
        self.site_name = "NeuroVT"
        self.openrouter_api_key = ""
        self.last_api_error = None
        
        self.sessions = {}
        self.current_session_id = None
        self.sessions_file = os.path.join(self.module_dir, "chats.json")
        
        self.load_settings()
        self.load_sessions()
        self.init_openrouter()
    
    def reset_all_settings(self):
        self.model = self.DEFAULT_MODEL
        self.temperature = self.DEFAULT_TEMPERATURE
        self.max_tokens = self.DEFAULT_MAX_TOKENS
        self.system_prompt = self.DEFAULT_SYSTEM_PROMPT
        self.fallback_responses = self.FALLBACK_RESPONSES.copy()
        
        self.save_module_settings({
            'openrouter_api_key': self.openrouter_api_key,
            'model': self.model,
            'temperature': self.temperature,
            'max_tokens': self.max_tokens,
            'system_prompt': self.system_prompt,
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
        self.openrouter_api_key = settings.get('openrouter_api_key', '')
        
        saved_fallback = settings.get('fallback_responses', [])
        if saved_fallback:
            self.fallback_responses = saved_fallback
        
        current_session = settings.get('current_session_id')
        if current_session:
            self.current_session_id = current_session
        
        print(f"[LLM] Загружены настройки: модель={self.model}, max_tokens={self.max_tokens}")
    
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
        
        if 'fallback_responses' in data and data['fallback_responses']:
            self.fallback_responses = [r.strip() for r in data['fallback_responses'] if r.strip()]
        
        self.save_module_settings({
            'openrouter_api_key': self.openrouter_api_key,
            'model': self.model,
            'temperature': self.temperature,
            'max_tokens': self.max_tokens,
            'system_prompt': self.system_prompt,
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
        
        if len(self.sessions[session_id]['messages']) > 100:
            self.sessions[session_id]['messages'] = self.sessions[session_id]['messages'][-100:]
        
        self.save_sessions()
        return True
    
    def clear_session_messages(self, session_id):
        if session_id in self.sessions:
            self.sessions[session_id]['messages'] = []
            self.sessions[session_id]['updated_at'] = datetime.now().isoformat()
            self.save_sessions()
            return True
        return False
    
    def get_session_context(self, session_id, max_messages=30):
        if session_id not in self.sessions:
            return []
        messages = self.sessions[session_id]['messages'][-max_messages:]
        print(f"[LLM] Контекст сессии {session_id}: {len(messages)} сообщений")
        for msg in messages[-3:]:
            print(f"[LLM]   {msg['role']}: {msg['content'][:50]}...")
        return messages
    
    def generate_with_ai(self, message, session_id=None):
        if not self.client:
            print(f"[LLM] Нет клиента OpenAI")
            return None
        
        target_session = session_id or self.current_session_id
        if not target_session or target_session not in self.sessions:
            print(f"[LLM] Сессия не найдена: {target_session}")
            return None
        
        try:
            context = self.get_session_context(target_session, 30)
            
            messages = [{"role": "system", "content": self.system_prompt}]
            
            for msg in context:
                messages.append({"role": msg['role'], "content": msg['content']})
            
            messages.append({"role": "user", "content": message})
            
            print(f"[LLM] Отправляем {len(messages)} сообщений в API (включая system и контекст)")
            
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                extra_headers={
                    "HTTP-Referer": self.site_url,
                    "X-Title": self.site_name,
                }
            )
            
            response = completion.choices[0].message.content.strip()
            print(f"[LLM] Получен ответ: {response[:100]}...")
            
            self.last_api_error = None
            return response
            
        except Exception as e:
            error_msg = str(e)
            self.last_api_error = error_msg
            print(f"[LLM] Ошибка API: {error_msg}")
            
            if "404" in error_msg:
                fallback_model = "arcee-ai/trinity-mini:free"
                if self.model != fallback_model:
                    print(f"[LLM] Пробуем fallback модель: {fallback_model}")
                    try:
                        context = self.get_session_context(target_session, 30)
                        messages = [{"role": "system", "content": self.system_prompt}]
                        for msg in context:
                            messages.append({"role": msg['role'], "content": msg['content']})
                        messages.append({"role": "user", "content": message})
                        
                        completion = self.client.chat.completions.create(
                            model=fallback_model,
                            messages=messages,
                            max_tokens=self.max_tokens,
                            temperature=self.temperature,
                        )
                        self.last_api_error = None
                        return completion.choices[0].message.content.strip()
                    except Exception as e2:
                        print(f"[LLM] Fallback тоже не работает: {e2}")
                        self.last_api_error = str(e2)
            return None
    
    def generate_fallback(self, message):
        response = random.choice(self.fallback_responses)
        if len(message) > 30:
            response += f" (Про '{message[:30]}...')"
        return response
    
    def generate_response(self, message, session_id=None):
        ai_response = self.generate_with_ai(message, session_id)
        if ai_response:
            return ai_response
        print(f"[LLM] Используем fallback ответ")
        return self.generate_fallback(message)
    
    def register_routes(self):
        @self.app.route('/api/llm/chat', methods=['POST'])
        def llm_chat():
            data = request.json
            user_message = data.get('message', '')
            session_id = data.get('session_id', self.current_session_id)
            
            if not user_message:
                return jsonify({"error": "Пустое сообщение"}), 400
            
            print(f"[LLM] Запрос к сессии {session_id}: {user_message[:50]}...")
            
            self.add_message_to_session(session_id, 'user', user_message)
            
            response_text = self.generate_response(user_message, session_id)
            
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
                return jsonify({
                    'session_id': session_id,
                    'session_name': self.sessions[session_id]['name'],
                    'messages': self.sessions[session_id]['messages']
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
            
            print(f"[LLM] Голосовой ввод в сессию {session_id}: {voice_text}")
            
            self.add_message_to_session(session_id, 'user', voice_text)
            
            response_text = self.generate_response(voice_text, session_id)
            
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
                "timestamp": datetime.now().isoformat()
            })

    def register_main_tab(self):
        return ("Чат с AI", """
        <div class="row" id="llm-module-container">
            <div class="col-md-4">
                <div class="card mb-3">
                    <div class="card-header">
                        <i class="fas fa-comments me-2"></i>
                        Сессии чатов
                        <button class="btn btn-sm btn-primary float-end" onclick="llmCreateNewSession()">
                            <i class="fas fa-plus"></i>
                        </button>
                    </div>
                    <div class="card-body" style="max-height: 500px; overflow-y: auto; padding: 0;">
                        <div id="llmSessionsList" class="list-group list-group-flush">
                            <div class="text-center p-3">Загрузка...</div>
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="col-md-8">
                <div class="card mb-3">
                    <div class="card-header d-flex justify-content-between align-items-center">
                        <span>
                            <i class="fas fa-comment me-2"></i>
                            <span id="llmCurrentSessionName">Загрузка...</span>
                        </span>
                        <div>
                            <span class="badge bg-info me-2" id="llmCurrentModelBadge">Загрузка...</span>
                            <span class="badge bg-secondary me-2" id="llmContextSizeBadge">0 сообщений</span>
                            <button class="btn btn-sm btn-outline-danger" onclick="llmClearCurrentSession()" title="Очистить историю">
                                <i class="fas fa-trash"></i>
                            </button>
                        </div>
                    </div>
                    <div class="card-body">
                        <div id="llmApiErrorAlert" class="alert alert-danger mb-3" style="display: none;">
                            <i class="fas fa-exclamation-triangle me-2"></i>
                            <span id="llmApiErrorMessage"></span>
                        </div>
                        
                        <div class="chat-log" id="llmChatMessages" style="height: 400px; overflow-y: auto;">
                            <div class="text-center text-muted p-3">Выберите сессию или создайте новую</div>
                        </div>
                        
                        <div class="mt-3">
                            <div class="input-group">
                                <textarea class="form-control" id="llmMessageInput" rows="2" 
                                          placeholder="Напишите сообщение... (Enter - отправить)"
                                          style="resize: none;"></textarea>
                                <button class="btn btn-primary" onclick="llmSendMessage()">
                                    <i class="fas fa-paper-plane"></i> Отправить
                                </button>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <script>
            let llmCurrentSessionId = null;
            let llmSessions = [];
            let llmCurrentModel = null;
            
            async function llmLoadCurrentModel() {
                try {
                    const response = await fetch('/api/llm/get_settings');
                    const data = await response.json();
                    llmCurrentModel = data.model;
                    const modelName = data.models[data.model] || data.model;
                    document.getElementById('llmCurrentModelBadge').innerHTML = '<i class="fas fa-brain me-1"></i>' + modelName;
                } catch(e) {
                    console.error('Ошибка загрузки модели:', e);
                }
            }
            
            async function llmLoadSessions() {
                try {
                    const response = await fetch('/api/llm/sessions');
                    const data = await response.json();
                    llmSessions = data.sessions;
                    llmCurrentSessionId = data.current_id;
                    
                    const sessionsList = document.getElementById('llmSessionsList');
                    if (llmSessions.length === 0) {
                        sessionsList.innerHTML = '<div class="text-center text-muted p-3">Нет сессий</div>';
                        return;
                    }
                    
                    sessionsList.innerHTML = llmSessions.map(session => `
                        <div class="list-group-item list-group-item-action ${session.is_current ? 'active' : ''}" 
                             style="background: ${session.is_current ? 'rgba(99, 102, 241, 0.2)' : 'transparent'};
                                    cursor: pointer; border-left: 3px solid ${session.is_current ? '#6366f1' : 'transparent'};">
                            <div class="d-flex justify-content-between align-items-center">
                                <div onclick="llmSwitchSession('${session.id}')" style="flex: 1; cursor: pointer;">
                                    <i class="fas fa-comment me-2"></i>
                                    <strong id="llmName-${session.id}">${llmEscapeHtml(session.name)}</strong>
                                    <br>
                                    <small class="text-muted">${session.message_count} сообщений</small>
                                </div>
                                <div>
                                    <button class="btn btn-sm btn-outline-secondary me-1" onclick="llmRenameSession('${session.id}')" title="Переименовать">
                                        <i class="fas fa-pencil-alt"></i>
                                    </button>
                                    <button class="btn btn-sm btn-outline-danger" onclick="llmDeleteSession('${session.id}')" title="Удалить">
                                        <i class="fas fa-times"></i>
                                    </button>
                                </div>
                            </div>
                        </div>
                    `).join('');
                    
                    const currentSession = llmSessions.find(s => s.id === llmCurrentSessionId);
                    if (currentSession) {
                        document.getElementById('llmCurrentSessionName').innerHTML = currentSession.name;
                        document.getElementById('llmContextSizeBadge').innerHTML = currentSession.message_count + ' сообщений';
                        await llmLoadSessionMessages(llmCurrentSessionId);
                    }
                    
                    await llmLoadCurrentModel();
                } catch(e) {
                    console.error('Ошибка загрузки сессий:', e);
                }
            }
            
            async function llmLoadSessionMessages(sessionId) {
                try {
                    const response = await fetch(`/api/llm/sessions/messages/${sessionId}`);
                    const data = await response.json();
                    
                    const chatMessages = document.getElementById('llmChatMessages');
                    if (data.messages.length === 0) {
                        chatMessages.innerHTML = '<div class="text-center text-muted p-3">Нет сообщений. Напишите что-нибудь!</div>';
                        return;
                    }
                    
                    chatMessages.innerHTML = data.messages.map(msg => `
                        <div class="chat-message ${msg.role === 'user' ? 'user' : 'ai'}">
                            <div class="d-flex align-items-start">
                                <div class="me-2">
                                    <i class="fas fa-${msg.role === 'user' ? 'user' : 'robot'}"></i>
                                </div>
                                <div style="flex: 1;">
                                    <small class="text-secondary">
                                        ${msg.role === 'user' ? 'Вы' : 'AI'}
                                        <span style="font-size: 0.7rem;">${new Date(msg.timestamp).toLocaleTimeString()}</span>
                                    </small>
                                    <div class="mt-1">${llmEscapeHtml(msg.content)}</div>
                                </div>
                            </div>
                        </div>
                    `).join('');
                    
                    chatMessages.scrollTop = chatMessages.scrollHeight;
                } catch(e) {
                    console.error('Ошибка загрузки сообщений:', e);
                }
            }
            
            function llmShowApiError(message) {
                const alertDiv = document.getElementById('llmApiErrorAlert');
                const messageSpan = document.getElementById('llmApiErrorMessage');
                messageSpan.innerHTML = message;
                alertDiv.style.display = 'block';
                setTimeout(() => {
                    alertDiv.style.display = 'none';
                }, 5000);
            }
            
            async function llmSendMessage() {
                const messageInput = document.getElementById('llmMessageInput');
                const message = messageInput.value.trim();
                if (!message) return;
                
                if (!llmCurrentSessionId) {
                    await llmCreateNewSession();
                }
                
                llmAddMessageToChat(message, 'user');
                messageInput.value = '';
                llmShowTypingIndicator();
                
                try {
                    const response = await fetch('/api/llm/chat', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({message: message, session_id: llmCurrentSessionId})
                    });
                    
                    const data = await response.json();
                    llmHideTypingIndicator();
                    
                    if (data.api_error) {
                        llmShowApiError('Ошибка API: ' + data.api_error + '. Использован fallback ответ.');
                    }
                    
                    llmAddMessageToChat(data.response, 'ai');
                    
                    if (data.context_size) {
                        document.getElementById('llmContextSizeBadge').innerHTML = data.context_size + ' сообщений';
                    }
                    
                    if (data.model_used !== llmCurrentModel) {
                        await llmLoadCurrentModel();
                    }
                    
                    await llmLoadSessions();
                } catch(error) {
                    llmHideTypingIndicator();
                    llmAddMessageToChat('Ошибка: ' + error.message, 'ai');
                    llmShowApiError('Ошибка соединения: ' + error.message);
                }
            }
            
            function llmAddMessageToChat(text, sender) {
                const chatMessages = document.getElementById('llmChatMessages');
                
                if (chatMessages.innerHTML.includes('Нет сообщений')) {
                    chatMessages.innerHTML = '';
                }
                
                const messageDiv = document.createElement('div');
                messageDiv.className = `chat-message ${sender}`;
                messageDiv.style.marginBottom = '1rem';
                messageDiv.innerHTML = `
                    <div class="d-flex align-items-start">
                        <div class="me-2"><i class="fas fa-${sender === 'user' ? 'user' : 'robot'}"></i></div>
                        <div style="flex: 1;">
                            <small class="text-secondary">${sender === 'user' ? 'Вы' : 'AI'}</small>
                            <div class="mt-1">${llmEscapeHtml(text)}</div>
                        </div>
                    </div>
                `;
                chatMessages.appendChild(messageDiv);
                chatMessages.scrollTop = chatMessages.scrollHeight;
            }
            
            async function llmCreateNewSession() {
                const name = prompt('Введите название сессии:', 'Новый чат');
                if (!name) return;
                
                const response = await fetch('/api/llm/sessions/create', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({name: name})
                });
                
                const data = await response.json();
                if (data.status === 'ok') {
                    await llmLoadSessions();
                    llmShowNotification('Сессия создана', 'success');
                }
            }
            
            async function llmSwitchSession(sessionId) {
                const response = await fetch(`/api/llm/sessions/switch/${sessionId}`, {
                    method: 'POST'
                });
                
                const data = await response.json();
                if (data.status === 'ok') {
                    llmCurrentSessionId = sessionId;
                    await llmLoadSessions();
                    llmShowNotification('Сессия переключена', 'success');
                }
            }
            
            async function llmRenameSession(sessionId) {
                const session = llmSessions.find(s => s.id === sessionId);
                const newName = prompt('Введите новое название:', session.name);
                if (!newName || newName === session.name) return;
                
                const response = await fetch(`/api/llm/sessions/rename/${sessionId}`, {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({name: newName})
                });
                
                const data = await response.json();
                if (data.status === 'ok') {
                    await llmLoadSessions();
                    llmShowNotification('Сессия переименована', 'success');
                }
            }
            
            async function llmDeleteSession(sessionId) {
                if (!confirm('Удалить эту сессию?')) return;
                
                const response = await fetch(`/api/llm/sessions/delete/${sessionId}`, {
                    method: 'DELETE'
                });
                
                const data = await response.json();
                if (data.status === 'ok') {
                    await llmLoadSessions();
                    llmShowNotification('Сессия удалена', 'success');
                }
            }
            
            async function llmClearCurrentSession() {
                if (!llmCurrentSessionId) return;
                if (!confirm('Очистить всю историю сообщений в этой сессии?')) return;
                
                const response = await fetch(`/api/llm/sessions/clear/${llmCurrentSessionId}`, {
                    method: 'POST'
                });
                
                const data = await response.json();
                if (data.status === 'ok') {
                    await llmLoadSessionMessages(llmCurrentSessionId);
                    document.getElementById('llmContextSizeBadge').innerHTML = '0 сообщений';
                    llmShowNotification('История очищена', 'success');
                }
            }
            
            function llmShowTypingIndicator() {
                const existing = document.getElementById('llmTypingIndicator');
                if (existing) existing.remove();
                
                const chatMessages = document.getElementById('llmChatMessages');
                const indicator = document.createElement('div');
                indicator.id = 'llmTypingIndicator';
                indicator.className = 'chat-message ai';
                indicator.innerHTML = `<div class="d-flex"><div class="spinner-border spinner-border-sm me-2"></div><span>Печатает...</span></div>`;
                chatMessages.appendChild(indicator);
                chatMessages.scrollTop = chatMessages.scrollHeight;
            }
            
            function llmHideTypingIndicator() {
                const indicator = document.getElementById('llmTypingIndicator');
                if (indicator) indicator.remove();
            }
            
            function llmEscapeHtml(text) {
                const div = document.createElement('div');
                div.textContent = text;
                return div.innerHTML;
            }
            
            function llmShowNotification(message, type) {
                const notification = document.createElement('div');
                notification.className = `alert alert-${type} position-fixed top-0 end-0 m-3`;
                notification.style.zIndex = '9999';
                notification.style.animation = 'fadeIn 0.3s ease';
                notification.style.background = type === 'success' ? '#10b981' : '#ef4444';
                notification.style.color = '#ffffff';
                notification.style.padding = '10px 20px';
                notification.style.borderRadius = '10px';
                notification.innerHTML = message;
                document.body.appendChild(notification);
                setTimeout(() => notification.remove(), 2000);
            }
            
            const llmMessageInput = document.getElementById('llmMessageInput');
            llmMessageInput.addEventListener('keydown', (e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    llmSendMessage();
                }
            });
            
            window.llmCreateNewSession = llmCreateNewSession;
            window.llmSwitchSession = llmSwitchSession;
            window.llmRenameSession = llmRenameSession;
            window.llmDeleteSession = llmDeleteSession;
            window.llmClearCurrentSession = llmClearCurrentSession;
            window.llmSendMessage = llmSendMessage;
            
            llmLoadSessions();
        </script>
        """)
    
    def register_settings_ui(self):
        return """
        <div class="row" id="llm-settings-container">
            <div class="col-md-12">
                <div class="alert alert-info mb-3" id="llmStatusAlert">
                    <i class="fas fa-info-circle me-2"></i>
                    <span id="llmStatusMessage">Загрузка настроек...</span>
                </div>
            </div>
            
            <div class="col-md-6">
                <div class="card mb-3">
                    <div class="card-header">API Настройки</div>
                    <div class="card-body">
                        <div class="mb-3">
                            <label class="form-label">OpenRouter API Key</label>
                            <input type="password" class="form-control" id="llmApiKeyInput" placeholder="sk-or-v1-...">
                            <small class="text-muted d-block mt-2">
                                <a href="https://openrouter.ai/keys" target="_blank">Получить ключ на OpenRouter</a>
                            </small>
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="col-md-6">
                <div class="card mb-3">
                    <div class="card-header">Модель и параметры</div>
                    <div class="card-body">
                        <div class="mb-3">
                            <label class="form-label">Модель AI</label>
                            <select class="form-select" id="llmModelSelect">
                                <option value="">Загрузка...</option>
                            </select>
                        </div>
                        
                        <div class="mb-3">
                            <label class="form-label">Температура: <span id="llmTemperatureValue">0.8</span></label>
                            <input type="range" class="form-range" id="llmTemperatureSlider" min="0" max="2" step="0.1" value="0.8">
                            <div class="d-flex justify-content-between">
                                <small>Точный</small>
                                <small>Креативный</small>
                            </div>
                        </div>
                        
                        <div class="mb-3">
                            <label class="form-label">Max Tokens (макс. длина ответа)</label>
                            <input type="number" class="form-control" id="llmMaxTokensInput" value="500" min="50" max="2000">
                            <small class="text-muted">Больше токенов = более длинные ответы</small>
                        </div>
                        
                        <div class="mb-3">
                            <label class="form-label">System Prompt</label>
                            <textarea class="form-control" id="llmSystemPromptInput" rows="3"></textarea>
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="col-md-12">
                <div class="card mb-3">
                    <div class="card-header">Fallback ответы (оффлайн режим)</div>
                    <div class="card-body">
                        <div class="alert alert-warning">
                            <i class="fas fa-exclamation-triangle me-2"></i>
                            Эти ответы используются, когда API недоступен.
                        </div>
                        
                        <div class="mb-3">
                            <textarea class="form-control" id="llmFallbackInput" rows="6" style="font-family: monospace;"></textarea>
                            <small class="text-muted">Каждый ответ с новой строки</small>
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="col-md-12">
                <div class="card">
                    <div class="card-body text-center">
                        <button class="btn btn-primary btn-lg" onclick="llmSaveAllSettings()">
                            <i class="fas fa-save me-2"></i>Сохранить все настройки
                        </button>
                        <button class="btn btn-outline-danger ms-3" onclick="llmResetAllSettings()">
                            <i class="fas fa-undo me-2"></i>Сбросить все настройки
                        </button>
                    </div>
                </div>
            </div>
        </div>
        
        <script>
            let llmCurrentSettings = {};
            
            async function llmLoadAllSettings() {
                try {
                    const response = await fetch('/api/llm/get_settings');
                    const data = await response.json();
                    llmCurrentSettings = data;
                    
                    const modelSelect = document.getElementById('llmModelSelect');
                    modelSelect.innerHTML = '';
                    for (const [id, name] of Object.entries(data.models)) {
                        const option = document.createElement('option');
                        option.value = id;
                        option.textContent = name;
                        if (id === data.model) {
                            option.selected = true;
                        }
                        modelSelect.appendChild(option);
                    }
                    
                    document.getElementById('llmTemperatureSlider').value = data.temperature;
                    document.getElementById('llmTemperatureValue').innerHTML = data.temperature;
                    document.getElementById('llmMaxTokensInput').value = data.max_tokens;
                    document.getElementById('llmSystemPromptInput').value = data.system_prompt;
                    document.getElementById('llmFallbackInput').value = data.fallback_responses.join('\\n');
                    
                    const statusMsg = document.getElementById('llmStatusMessage');
                    if (data.has_api_key) {
                        statusMsg.innerHTML = '✅ API ключ настроен. Текущая модель: ' + (data.models[data.model] || data.model);
                        statusMsg.parentElement.className = 'alert alert-success mb-3';
                    } else {
                        statusMsg.innerHTML = '⚠️ API ключ не настроен. Будут использоваться fallback ответы.';
                        statusMsg.parentElement.className = 'alert alert-warning mb-3';
                    }
                } catch(e) {
                    console.error('Ошибка загрузки:', e);
                    document.getElementById('llmStatusMessage').innerHTML = '❌ Ошибка загрузки настроек';
                }
            }
            
            async function llmSaveAllSettings() {
                const apiKey = document.getElementById('llmApiKeyInput').value;
                const model = document.getElementById('llmModelSelect').value;
                const temperature = parseFloat(document.getElementById('llmTemperatureSlider').value);
                const maxTokens = parseInt(document.getElementById('llmMaxTokensInput').value);
                const systemPrompt = document.getElementById('llmSystemPromptInput').value;
                const fallbackText = document.getElementById('llmFallbackInput').value;
                const fallbackResponses = fallbackText.split('\\n').filter(l => l.trim().length > 0);
                
                if (fallbackResponses.length === 0) {
                    llmShowNotification('Добавьте хотя бы один fallback ответ', 'error');
                    return;
                }
                
                const saveData = {
                    openrouter_api_key: apiKey,
                    model: model,
                    temperature: temperature,
                    max_tokens: maxTokens,
                    system_prompt: systemPrompt,
                    fallback_responses: fallbackResponses
                };
                
                const response = await fetch('/api/llm/save_all_settings', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(saveData)
                });
                
                const data = await response.json();
                if (data.status === 'ok') {
                    llmShowNotification('✅ Все настройки сохранены! Модель: ' + (llmCurrentSettings.models[model] || model), 'success');
                    document.getElementById('llmApiKeyInput').value = '';
                    llmLoadAllSettings();
                    
                    const modelBadge = document.getElementById('llmCurrentModelBadge');
                    if (modelBadge && window.location.pathname === '/') {
                        modelBadge.innerHTML = '<i class="fas fa-brain me-1"></i>' + (llmCurrentSettings.models[model] || model);
                    }
                } else {
                    llmShowNotification('❌ Ошибка сохранения', 'error');
                }
            }
            
            async function llmResetAllSettings() {
                if (!confirm('Сбросить ВСЕ настройки (модель, температуру, max tokens, system prompt и fallback ответы) к значениям по умолчанию? API ключ не будет удалён.')) return;
                
                const response = await fetch('/api/llm/reset_settings', {
                    method: 'POST'
                });
                
                const data = await response.json();
                if (data.status === 'ok') {
                    llmShowNotification('✅ Все настройки сброшены к значениям по умолчанию', 'success');
                    await llmLoadAllSettings();
                    
                    const modelSelect = document.getElementById('llmModelSelect');
                    if (modelSelect) {
                        for (let i = 0; i < modelSelect.options.length; i++) {
                            if (modelSelect.options[i].value === data.default_model) {
                                modelSelect.selectedIndex = i;
                                break;
                            }
                        }
                    }
                    
                    document.getElementById('llmTemperatureSlider').value = data.default_temperature;
                    document.getElementById('llmTemperatureValue').innerHTML = data.default_temperature;
                    document.getElementById('llmMaxTokensInput').value = data.default_max_tokens;
                    document.getElementById('llmSystemPromptInput').value = data.default_system_prompt;
                } else {
                    llmShowNotification('❌ Ошибка сброса настроек', 'error');
                }
            }
            
            document.getElementById('llmTemperatureSlider').addEventListener('input', function() {
                document.getElementById('llmTemperatureValue').innerHTML = this.value;
            });
            
            function llmShowNotification(message, type) {
                const notification = document.createElement('div');
                notification.className = `alert alert-${type} position-fixed top-0 end-0 m-3`;
                notification.style.zIndex = '9999';
                notification.style.animation = 'fadeIn 0.3s ease';
                notification.style.background = type === 'success' ? '#10b981' : (type === 'error' ? '#ef4444' : '#3b82f6');
                notification.style.color = '#ffffff';
                notification.style.padding = '10px 20px';
                notification.style.borderRadius = '10px';
                notification.innerHTML = message;
                document.body.appendChild(notification);
                setTimeout(() => notification.remove(), 3000);
            }
            
            window.llmSaveAllSettings = llmSaveAllSettings;
            window.llmResetAllSettings = llmResetAllSettings;
            
            llmLoadAllSettings();
        </script>
        """
    
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
        
        print(f"[LLM] Обработка голоса из STT: {text}")
        
        self.add_message_to_session(session_id, 'user', text)
        
        response_text = self.generate_response(text, session_id)
        
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