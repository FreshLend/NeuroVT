import json
import os

class BaseModule:
    name = "base_module"
    display_name = "Базовый модуль"
    
    def __init__(self, app, event_bus, socketio):
        self.app = app
        self.event_bus = event_bus
        self.socketio = socketio
        self.module_dir = os.path.join("modules", self.__class__.__name__.replace("Module", "_Module"))
        self.settings_file = os.path.join(self.module_dir, "settings.json")
        self.templates_dir = os.path.join(self.module_dir, "templates")
    
    def load_module_settings(self):
        if os.path.exists(self.settings_file):
            try:
                with open(self.settings_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def save_module_settings(self, settings):
        os.makedirs(self.module_dir, exist_ok=True)
        try:
            current = self.load_module_settings()
            current.update(settings)
            with open(self.settings_file, 'w', encoding='utf-8') as f:
                json.dump(current, f, indent=2, ensure_ascii=False)
            return True
        except:
            return False
    
    def get_template_content(self, template_name):
        template_path = os.path.join(self.templates_dir, template_name)
        if os.path.exists(template_path):
            try:
                with open(template_path, 'r', encoding='utf-8') as f:
                    return f.read()
            except Exception as e:
                print(f"[{self.display_name}] Ошибка чтения шаблона {template_name}: {e}")
                return ""
        return ""
    
    def register_routes(self):
        pass
    
    def register_socketio_handlers(self, sio):
        pass
    
    def register_settings_ui(self):
        return self.get_template_content("settings.html")
    
    def register_main_tab(self):
        content = self.get_template_content("main_tab.html")
        if content:
            return (self.display_name, content)
        return None
    
    def on_load(self):
        pass