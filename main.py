import json
import os
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO
from core.event_bus import event_bus
from core.module_loader import load_modules

app = Flask(__name__)
app.config['SECRET_KEY'] = 'vtuber_secret_key'
socketio = SocketIO(app, cors_allowed_origins="*")

main_tabs = []
settings_tabs = []

def load_global_settings():
    settings = {}
    modules_dir = "modules"
    if os.path.exists(modules_dir):
        for module_folder in os.listdir(modules_dir):
            module_settings_path = os.path.join(modules_dir, module_folder, "settings.json")
            if os.path.exists(module_settings_path):
                try:
                    with open(module_settings_path, 'r', encoding='utf-8') as f:
                        module_settings = json.load(f)
                        settings[module_folder.lower().replace('_module', '')] = module_settings
                except:
                    pass
    return settings

app.config['SETTINGS'] = load_global_settings()

@app.route('/')
def index():
    return render_template('index.html', tabs=main_tabs)

@app.route('/settings')
def settings_page():
    return render_template('settings.html', settings_tabs=settings_tabs)

@app.route('/api/save_all_settings', methods=['POST'])
def save_all_settings():
    try:
        data = request.json
        modules_dir = "modules"
        for module_name, module_settings in data.items():
            for module_folder in os.listdir(modules_dir):
                settings_path = os.path.join(modules_dir, module_folder, "settings.json")
                if os.path.exists(settings_path):
                    try:
                        with open(settings_path, 'r', encoding='utf-8') as f:
                            current = json.load(f)
                        current.update(module_settings)
                        with open(settings_path, 'w', encoding='utf-8') as f:
                            json.dump(current, f, indent=2, ensure_ascii=False)
                    except:
                        pass
        app.config['SETTINGS'] = load_global_settings()
        return jsonify({"status": "ok", "message": "Все настройки сохранены"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/get_all_settings', methods=['GET'])
def get_all_settings():
    return jsonify(app.config['SETTINGS'])

def main():
    global main_tabs, settings_tabs
    
    modules = load_modules(app, event_bus, socketio)
    
    if modules:
        print(f"\nЗагружено модулей: {len(modules)}")
        for mod in modules:
            print(f"  {mod.display_name}")
            mod.register_routes()
            mod.register_socketio_handlers(socketio)
            
            settings_ui = mod.register_settings_ui()
            if settings_ui:
                settings_tabs.append((mod.name, mod.display_name, settings_ui))
            
            main_tab = mod.register_main_tab()
            if main_tab:
                main_tabs.append((mod.name, main_tab[0], main_tab[1]))
        
        print("\nАктивация модулей...")
        for mod in modules:
            mod.on_load()
    
    print("\n" + "=" * 60)
    print("Сервер запущен: http://localhost:5000")
    print("=" * 60 + "\n")
    
    socketio.run(app, debug=True, host='0.0.0.0', port=5000, allow_unsafe_werkzeug=True)

if __name__ == '__main__':
    main()