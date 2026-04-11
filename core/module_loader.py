import os
import importlib
import sys

def load_modules(app, event_bus, modules_dir="modules"):
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
        
        module_files = [f for f in os.listdir(folder_path) if f.endswith('.py') and f != '__init__.py']
        
        for filename in module_files:
            if filename.endswith('_module.py'):
                try:
                    module_name = filename[:-3]
                    spec = importlib.util.spec_from_file_location(
                        f"modules.{folder_name}.{module_name}",
                        os.path.join(folder_path, filename)
                    )
                    if spec and spec.loader:
                        mod = importlib.util.module_from_spec(spec)
                        sys.modules[f"modules.{folder_name}.{module_name}"] = mod
                        spec.loader.exec_module(mod)
                        
                        from modules.base_module import BaseModule
                        for attr_name in dir(mod):
                            attr = getattr(mod, attr_name)
                            if (isinstance(attr, type) and issubclass(attr, BaseModule) 
                                and attr != BaseModule):
                                instance = attr(app, event_bus)
                                modules.append(instance)
                                print(f"[LOADER] Загружен: {instance.display_name}")
                except Exception as e:
                    print(f"[LOADER] Ошибка {filename}: {e}")
    
    return modules