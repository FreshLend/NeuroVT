import os
import importlib.util
import sys
import config

def load_modules(app, event_bus, socketio, modules_dir="modules"):
    modules = []
    
    if not os.path.exists(modules_dir):
        os.makedirs(modules_dir)
        print(f"[LOADER] Создана папка {modules_dir}")
        return modules
    
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    for folder_name in os.listdir(modules_dir):
        folder_path = os.path.join(modules_dir, folder_name)
        if not os.path.isdir(folder_path):
            continue
        
        if folder_name in config.DISABLED_MODULES:
            print(f"[LOADER] Модуль отключен: {folder_name}")
            continue
        
        main_file = os.path.join(folder_path, "main.py")
        if not os.path.exists(main_file):
            continue
        
        try:
            spec = importlib.util.spec_from_file_location(
                f"modules.{folder_name}.main",
                main_file
            )
            if spec and spec.loader:
                mod = importlib.util.module_from_spec(spec)
                sys.modules[f"modules.{folder_name}.main"] = mod
                spec.loader.exec_module(mod)
                
                from modules.base_module import BaseModule
                module_classes = []
                for attr_name in dir(mod):
                    attr = getattr(mod, attr_name)
                    if (isinstance(attr, type) and issubclass(attr, BaseModule) 
                        and attr != BaseModule):
                        module_classes.append(attr)
                
                if module_classes:
                    module_class = module_classes[0]
                    instance = module_class(app, event_bus, socketio)
                    modules.append(instance)
                    print(f"[LOADER] Загружен: {instance.display_name} (из {folder_name})")
        except Exception as e:
            print(f"[LOADER] Ошибка загрузки модуля {folder_name}: {e}")
    
    return modules
