class EventBus:
    def __init__(self):
        self._listeners = {}
    
    def subscribe(self, event_type, callback):
        if event_type not in self._listeners:
            self._listeners[event_type] = []
        self._listeners[event_type].append(callback)
    
    def emit(self, event_type, data=None):
        if event_type in self._listeners:
            for callback in self._listeners[event_type]:
                callback(data)
    
    def on(self, event_type):
        def decorator(func):
            self.subscribe(event_type, func)
            return func
        return decorator

event_bus = EventBus()