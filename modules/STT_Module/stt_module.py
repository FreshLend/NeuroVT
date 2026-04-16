import threading
import queue
import time
import pyaudio
import keyboard
import numpy as np
from modules.base_module import BaseModule
from flask import jsonify, request
from datetime import datetime
from faster_whisper import WhisperModel

class STTModule(BaseModule):
    name = "stt"
    display_name = "STT (Faster-Whisper)"
    
    def __init__(self, app, event_bus, socketio):
        super().__init__(app, event_bus, socketio)
        self.model = None
        self.model_loaded = False
        self.is_listening = False
        self.audio_queue = queue.Queue()
        self.listen_thread = None
        self.current_text = ""
        self.sample_rate = 16000
        self.chunk_size = 1024
        self.channels = 1
        self.format = pyaudio.paInt16
        self.model_size = "small"
        self.device = "cpu"
        self.compute_type = "int8"
        self.recognition_language = None
        self.audio_buffer = bytearray()
        self.is_speaking = False
        self.recognized_history = []
        self.device_index = None
        self.available_devices = []
        self.hotkey = "ctrl+shift+m"
        self.hotkey_listener_active = False
        
        self.load_settings()
        self.scan_audio_devices()
        threading.Thread(target=self.load_model, daemon=True).start()
        self.start_hotkey_listener()
    
    def load_settings(self):
        settings = self.load_module_settings()
        self.device_index = settings.get('device_index')
        self.model_size = settings.get('model_size', 'small')
        self.device = settings.get('device', 'cpu')
        self.compute_type = settings.get('compute_type', 'int8')
        self.hotkey = settings.get('hotkey', 'ctrl+shift+m')
        self.recognition_language = settings.get('recognition_language')
    
    def save_settings(self):
        self.save_module_settings({
            'device_index': self.device_index,
            'model_size': self.model_size,
            'device': self.device,
            'compute_type': self.compute_type,
            'hotkey': self.hotkey,
            'recognition_language': self.recognition_language
        })
    
    def scan_audio_devices(self):
        try:
            p = pyaudio.PyAudio()
            self.available_devices = []
            for i in range(p.get_device_count()):
                device_info = p.get_device_info_by_index(i)
                if device_info['maxInputChannels'] > 0:
                    self.available_devices.append({
                        'index': i,
                        'name': device_info['name'],
                        'channels': int(device_info['maxInputChannels']),
                        'sample_rate': int(device_info['defaultSampleRate'])
                    })
            p.terminate()
            if self.device_index is None and self.available_devices:
                self.device_index = self.available_devices[0]['index']
                self.save_settings()
        except Exception as e:
            print(f"[STT] Ошибка сканирования устройств: {e}")
    
    def load_model(self):
        try:
            print(f"[STT] Загрузка модели {self.model_size}...")
            self.model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
                cpu_threads=4,
                num_workers=1
            )
            self.model_loaded = True
            print(f"[STT] Модель загружена")
            return True
        except Exception as e:
            print(f"[STT] Ошибка загрузки модели: {e}")
            return False
    
    def start_hotkey_listener(self):
        if self.hotkey_listener_active:
            return
        try:
            keyboard.add_hotkey(self.hotkey, self.on_hotkey_pressed)
            self.hotkey_listener_active = True
            print(f"[STT] Горячая клавиша {self.hotkey} зарегистрирована")
        except Exception as e:
            print(f"[STT] Ошибка регистрации хоткея: {e}")
    
    def stop_hotkey_listener(self):
        if not self.hotkey_listener_active:
            return
        try:
            keyboard.remove_hotkey(self.hotkey)
            self.hotkey_listener_active = False
            print(f"[STT] Горячая клавиша {self.hotkey} удалена")
        except Exception as e:
            print(f"[STT] Ошибка удаления хоткея: {e}")
    
    def on_hotkey_pressed(self):
        if self.is_listening:
            self.stop_listening()
        else:
            self.start_listening()
    
    def listen_worker(self):
        p = None
        stream = None
        try:
            p = pyaudio.PyAudio()
            stream = p.open(
                format=self.format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                input_device_index=self.device_index,
                frames_per_buffer=self.chunk_size
            )
            audio_buffer = bytearray()
            last_speech_time = 0
            min_speech_duration = 0.8
            while self.is_listening:
                try:
                    data = stream.read(self.chunk_size, exception_on_overflow=False)
                    audio_buffer.extend(data)
                    audio_np = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
                    volume = np.sqrt(np.mean(audio_np**2))
                    is_speech = volume > 0.01
                    if is_speech:
                        last_speech_time = time.time()
                        if not self.is_speaking:
                            self.is_speaking = True
                    else:
                        if self.is_speaking and (time.time() - last_speech_time) > 0.8:
                            self.is_speaking = False
                            if len(audio_buffer) > self.sample_rate * min_speech_duration * 2:
                                threading.Thread(
                                    target=self.process_audio,
                                    args=(bytes(audio_buffer),),
                                    daemon=True
                                ).start()
                            audio_buffer = bytearray()
                    if len(audio_buffer) > self.sample_rate * 15 * 2:
                        audio_buffer = bytearray()
                except Exception as e:
                    print(f"[STT] Ошибка чтения: {e}")
                    time.sleep(0.1)
        except Exception as e:
            print(f"[STT] Ошибка воркера: {e}")
        finally:
            if stream:
                stream.stop_stream()
                stream.close()
            if p:
                p.terminate()
    
    def process_audio(self, audio_bytes):
        try:
            audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            language_param = self.recognition_language if self.recognition_language else None
            segments, info = self.model.transcribe(
                audio_np,
                beam_size=5,
                language=language_param,
                vad_filter=True,
                vad_parameters={"min_silence_duration_ms": 500, "speech_pad_ms": 400}
            )
            recognized_text = " ".join([segment.text for segment in segments]).strip()
            detected_lang = info.language
            if recognized_text:
                print(f"[STT] ✅ Распознано ({detected_lang}): {recognized_text}")
                self.current_text = recognized_text
                self.recognized_history.append({
                    "text": recognized_text,
                    "timestamp": datetime.now().isoformat(),
                    "language": detected_lang
                })
                if len(self.recognized_history) > 20:
                    self.recognized_history = self.recognized_history[-20:]
                self.event_bus.emit("stt_text_ready", {
                    "text": recognized_text,
                    "is_final": True,
                    "timestamp": datetime.now().isoformat(),
                    "language": detected_lang
                })
                self.event_bus.emit("llm_voice_input", {
                    "text": recognized_text,
                    "source": "microphone",
                    "timestamp": datetime.now().isoformat()
                })
        except Exception as e:
            print(f"[STT] Ошибка распознавания: {e}")
    
    def start_listening(self):
        if self.is_listening:
            return
        if not self.model_loaded:
            for _ in range(30):
                if self.model_loaded:
                    break
                time.sleep(1)
            if not self.model_loaded:
                print(f"[STT] Модель не загружена!")
                return
        self.is_listening = True
        self.listen_thread = threading.Thread(target=self.listen_worker, daemon=True)
        self.listen_thread.start()
    
    def stop_listening(self):
        if not self.is_listening:
            return
        self.is_listening = False
        if self.listen_thread:
            self.listen_thread.join(timeout=2)
        self.is_speaking = False
    
    def register_routes(self):
        @self.app.route('/api/stt/status', methods=['GET'])
        def stt_status():
            return jsonify({
                "model_loaded": self.model_loaded,
                "is_listening": self.is_listening,
                "current_text": self.current_text,
                "available_devices": self.available_devices,
                "current_device": self.device_index,
                "model_size": self.model_size,
                "device": self.device,
                "is_speaking": self.is_speaking,
                "hotkey": self.hotkey,
                "recognition_language": self.recognition_language,
                "history": self.recognized_history[-10:]
            })
        
        @self.app.route('/api/stt/get_settings', methods=['GET'])
        def stt_get_settings():
            return jsonify({
                "device_index": self.device_index,
                "available_devices": self.available_devices,
                "model_loaded": self.model_loaded,
                "is_listening": self.is_listening,
                "model_size": self.model_size,
                "device": self.device,
                "compute_type": self.compute_type,
                "hotkey": self.hotkey,
                "recognition_language": self.recognition_language
            })
        
        @self.app.route('/api/stt/set_device', methods=['POST'])
        def stt_set_device():
            data = request.json
            device_index = data.get('device_index')
            if device_index is not None:
                was_listening = self.is_listening
                if was_listening:
                    self.stop_listening()
                self.device_index = device_index
                self.save_settings()
                if was_listening:
                    time.sleep(0.5)
                    self.start_listening()
                return jsonify({"status": "ok", "device_index": device_index})
            return jsonify({"error": "Неверный индекс"}), 400
        
        @self.app.route('/api/stt/set_hotkey', methods=['POST'])
        def stt_set_hotkey():
            data = request.json
            hotkey = data.get('hotkey')
            if hotkey and hotkey.strip():
                self.stop_hotkey_listener()
                self.hotkey = hotkey.strip()
                self.save_settings()
                self.start_hotkey_listener()
                return jsonify({"status": "ok", "hotkey": self.hotkey})
            return jsonify({"error": "Неверная комбинация"}), 400
        
        @self.app.route('/api/stt/set_language', methods=['POST'])
        def stt_set_language():
            data = request.json
            language = data.get('language')
            if language == "auto" or language == "None" or language is None:
                self.recognition_language = None
            elif language in ["ru", "en", "fr", "de", "es", "it", "zh", "ja", "ko"]:
                self.recognition_language = language
            else:
                return jsonify({"error": "Неподдерживаемый язык"}), 400
            self.save_settings()
            return jsonify({"status": "ok", "recognition_language": self.recognition_language})
        
        @self.app.route('/api/stt/start', methods=['POST'])
        def stt_start():
            if not self.is_listening:
                self.start_listening()
                return jsonify({"status": "ok", "message": "Распознавание запущено"})
            return jsonify({"status": "ok", "message": "Уже работает"})
        
        @self.app.route('/api/stt/stop', methods=['POST'])
        def stt_stop():
            if self.is_listening:
                self.stop_listening()
                return jsonify({"status": "ok", "message": "Распознавание остановлено"})
            return jsonify({"status": "ok", "message": "Уже остановлено"})
        
        @self.app.route('/api/stt/restart', methods=['POST'])
        def stt_restart():
            self.stop_listening()
            time.sleep(0.5)
            self.start_listening()
            return jsonify({"status": "ok", "message": "Распознавание перезапущено"})
    
    def on_load(self):
        self.event_bus.subscribe("llm_voice_input", self.handle_voice_input)
    
    def handle_voice_input(self, data):
        pass