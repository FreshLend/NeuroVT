import os
from flask import render_template, jsonify, request, send_from_directory
from modules.base_module import BaseModule

class AvatarLive2D(BaseModule):
    name = "avatar_live2d"
    display_name = "Аватар (Live2D)"
    category = "Avatar"
    icon = "fa-circle-user"

    def __init__(self, app, event_bus, socketio):
        super().__init__(app, event_bus, socketio)
        if self.templates_dir not in app.jinja_loader.searchpath:
            app.jinja_loader.searchpath.append(self.templates_dir)

        self.settings = self.load_module_settings()
        self.model_name = self.settings.get('model_name', '')
        self.triggers_by_model = self.settings.get('triggers_by_model', {})
        self.emotion_duration = self.settings.get('emotion_duration', 3.0)
        self.emotion_random = self.settings.get('emotion_random', False)
        self.emotion_random_min = self.settings.get('emotion_random_min', 4.0)
        self.emotion_random_max = self.settings.get('emotion_random_max', 10.0)

        self.expressions_cache = []
        self.available_models = self.scan_models_folder()

        if self.model_name:
            self.load_expressions_from_model()

    def scan_models_folder(self):
        models_dir = os.path.join(self.module_dir, 'models')
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
            return []
        models = []
        for item in os.listdir(models_dir):
            item_path = os.path.join(models_dir, item)
            if os.path.isdir(item_path):
                for f in os.listdir(item_path):
                    if f.endswith('.model3.json'):
                        models.append(item)
                        break
        return models

    def get_model_url(self, model_name=None):
        name = model_name or self.model_name
        if not name:
            return None
        model_dir = os.path.join(self.module_dir, 'models', name)
        if not os.path.isdir(model_dir):
            return None
        for f in os.listdir(model_dir):
            if f.endswith('.model3.json'):
                return f'/avatar/live2d/model/{name}/{f}'
        return None

    def scan_expressions_folder(self, model_folder_path):
        expressions = []
        if not os.path.isdir(model_folder_path):
            return expressions
        for f in os.listdir(model_folder_path):
            if f.endswith('.exp3.json'):
                name = f[:-10]
                expressions.append(name)
        return expressions

    def load_expressions_from_model(self):
        model_folder = os.path.join(self.module_dir, 'models', self.model_name)
        if not os.path.isdir(model_folder):
            self.expressions_cache = []
            return
        self.expressions_cache = self.scan_expressions_folder(model_folder)

    def get_current_triggers(self):
        return self.triggers_by_model.get(self.model_name, {})

    def set_current_triggers(self, triggers):
        self.triggers_by_model[self.model_name] = triggers
        self.save_module_settings({
            'model_name': self.model_name,
            'triggers_by_model': self.triggers_by_model,
            'emotion_duration': self.emotion_duration,
            'emotion_random': self.emotion_random,
            'emotion_random_min': self.emotion_random_min,
            'emotion_random_max': self.emotion_random_max
        })
        self.socketio.emit('avatar_triggers_updated', {
            'model_name': self.model_name,
            'triggers': triggers,
            'emotion_duration': self.emotion_duration,
            'emotion_random': self.emotion_random,
            'emotion_random_min': self.emotion_random_min,
            'emotion_random_max': self.emotion_random_max
        })

    def set_model(self, model_name):
        if model_name not in self.available_models:
            return False
        self.model_name = model_name
        self.load_expressions_from_model()
        self.save_module_settings({
            'model_name': self.model_name,
            'triggers_by_model': self.triggers_by_model,
            'emotion_duration': self.emotion_duration,
            'emotion_random': self.emotion_random,
            'emotion_random_min': self.emotion_random_min,
            'emotion_random_max': self.emotion_random_max
        })
        self.socketio.emit('avatar_model_changed', {
            'model_name': self.model_name,
            'expressions': self.expressions_cache
        })
        self.socketio.emit('avatar_triggers_updated', {
            'model_name': self.model_name,
            'triggers': self.get_current_triggers(),
            'emotion_duration': self.emotion_duration,
            'emotion_random': self.emotion_random,
            'emotion_random_min': self.emotion_random_min,
            'emotion_random_max': self.emotion_random_max
        })
        return True

    def register_routes(self):
        @self.app.route('/avatar/live2d')
        def avatar_page():
            return render_template('avatar.html',
                                   model_url=self.get_model_url(),
                                   model_name=self.model_name)

        @self.app.route('/avatar/live2d/model/<path:filename>')
        def avatar_model_file(filename):
            parts = filename.split('/', 1)
            if len(parts) != 2:
                return "Invalid path", 400
            model_name, file_path = parts
            model_dir = os.path.join(self.module_dir, 'models', model_name)
            full_path = os.path.join(model_dir, file_path)
            if not os.path.realpath(full_path).startswith(os.path.realpath(model_dir)):
                return "Forbidden", 403
            if not os.path.exists(full_path):
                return "File not found", 404
            return send_from_directory(model_dir, file_path)

        @self.app.route('/avatar/live2d/static/<path:filename>')
        def avatar_static(filename):
            return send_from_directory(os.path.join(self.module_dir, 'static'), filename)

        @self.app.route('/api/avatar/live2d/get_models', methods=['GET'])
        def get_models():
            return jsonify({'models': self.available_models, 'current': self.model_name})

        @self.app.route('/api/avatar/live2d/set_model', methods=['POST'])
        def set_model():
            data = request.json
            new_model = data.get('model_name')
            if self.set_model(new_model):
                return jsonify({'status': 'ok', 'model_name': self.model_name, 'expressions': self.expressions_cache})
            return jsonify({'error': 'Модель не найдена'}), 400

        @self.app.route('/api/avatar/live2d/get_triggers', methods=['GET'])
        def get_triggers():
            return jsonify({
                'triggers': self.get_current_triggers(),
                'expressions': self.expressions_cache,
                'emotion_duration': self.emotion_duration,
                'emotion_random': self.emotion_random,
                'emotion_random_min': self.emotion_random_min,
                'emotion_random_max': self.emotion_random_max
            })

        @self.app.route('/api/avatar/live2d/save_triggers', methods=['POST'])
        def save_triggers():
            data = request.json
            new_triggers = data.get('triggers', {})
            cleaned = {}
            for expr, keywords in new_triggers.items():
                if isinstance(keywords, list):
                    cleaned[expr] = [kw.strip() for kw in keywords if kw.strip()]
                elif isinstance(keywords, str):
                    cleaned[expr] = [kw.strip() for kw in keywords.split(',') if kw.strip()]
                else:
                    cleaned[expr] = []
            self.set_current_triggers(cleaned)
            return jsonify({'status': 'ok'})

        @self.app.route('/api/avatar/live2d/save_emotion_settings', methods=['POST'])
        def save_emotion_settings():
            data = request.json
            self.emotion_duration = float(data.get('duration', 3.0))
            self.emotion_random = data.get('random', False)
            self.emotion_random_min = float(data.get('random_min', 4.0))
            self.emotion_random_max = float(data.get('random_max', 10.0))
            self.save_module_settings({
                'model_name': self.model_name,
                'triggers_by_model': self.triggers_by_model,
                'emotion_duration': self.emotion_duration,
                'emotion_random': self.emotion_random,
                'emotion_random_min': self.emotion_random_min,
                'emotion_random_max': self.emotion_random_max
            })
            self.socketio.emit('avatar_triggers_updated', {
                'model_name': self.model_name,
                'triggers': self.get_current_triggers(),
                'emotion_duration': self.emotion_duration,
                'emotion_random': self.emotion_random,
                'emotion_random_min': self.emotion_random_min,
                'emotion_random_max': self.emotion_random_max
            })
            return jsonify({'status': 'ok'})

    def register_socketio_handlers(self, sio):
        @sio.on('avatar_request_triggers')
        def handle_request_triggers(data):
            self.socketio.emit('avatar_triggers_updated', {
                'model_name': self.model_name,
                'triggers': self.get_current_triggers(),
                'emotion_duration': self.emotion_duration,
                'emotion_random': self.emotion_random,
                'emotion_random_min': self.emotion_random_min,
                'emotion_random_max': self.emotion_random_max
            }, room=request.sid)

    def register_main_tab(self):
        return ("Аватар", self.get_template_content("main_tab.html"))

    def register_settings_ui(self):
        return self.get_template_content("settings.html")

    def on_load(self):
        self.event_bus.subscribe("tts_speak", self.handle_tts)

    def handle_tts(self, data):
        text = data.get('text', '')
        if not text:
            return
        self.socketio.emit('avatar_tts_text', {'text': text})