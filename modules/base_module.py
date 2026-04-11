import json
import os

class BaseModule:
    name = "base_module"
    display_name = "Базовый модуль"
    
    def __init__(self, app, event_bus):
        self.app = app
        self.event_bus = event_bus
        self.module_dir = os.path.join("modules", self.__class__.__name__.replace("Module", "_Module"))
        self.settings_file = os.path.join(self.module_dir, "settings.json")
    
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
    
    def register_routes(self):
        pass
    
    def register_settings_ui(self):
        return None
    
    def register_main_tab(self):
        return None
    
    def on_load(self):
        pass