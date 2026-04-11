import json
import os
import threading
import queue
import time
import pyaudio
import vosk
from modules.base_module import BaseModule
from flask import jsonify, request
from datetime import datetime

class STTModule(BaseModule):
    name = "stt"
    display_name = "STT Распознавание речи (Vosk)"
    
    def __init__(self, app, event_bus):
        super().__init__(app, event_bus)
        self.model = None
        self.model_loaded = False
        self.rec = None
        self.is_listening = False
        self.audio_queue = queue.Queue()
        self.listen_thread = None
        self.current_text = ""
        
        self.sample_rate = 16000
        self.chunk_size = 4000
        self.channels = 1
        self.format = pyaudio.paInt16
        
        self.model_name = "vosk-model-small-ru-0.22"
        self.model_path = os.path.join(self.module_dir, "models", self.model_name)
        
        self.device_index = None
        self.available_devices = []
        
        self.load_settings()
        self.scan_audio_devices()
        self.load_model()
    
    def load_settings(self):
        settings = self.load_module_settings()
        self.device_index = settings.get('device_index')
        print(f"[STT] Загружены настройки")
    
    def save_settings(self):
        self.save_module_settings({
            'device_index': self.device_index
        })
        print(f"[STT] Настройки сохранены")
    
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
            print(f"[STT] Найдено {len(self.available_devices)} устройств ввода")
            for dev in self.available_devices:
                print(f"[STT]   {dev['index']}: {dev['name']}")
        except Exception as e:
            print(f"[STT] Ошибка сканирования устройств: {e}")
    
    def load_model(self):
        if not os.path.exists(self.model_path):
            print(f"[STT] Модель не найдена: {self.model_path}")
            print(f"[STT] Скачайте модель с https://alphacephei.com/vosk/models")
            print(f"[STT] И распакуйте в {self.model_path}")
            return False
        
        try:
            print(f"[STT] Загрузка модели {self.model_name}...")
            self.model = vosk.Model(self.model_path)
            self.model_loaded = True
            print(f"[STT] Модель загружена")
            return True
        except Exception as e:
            print(f"[STT] Ошибка загрузки модели: {e}")
            return False
    
    def init_recognizer(self):
        if not self.model_loaded:
            return False
        
        try:
            self.rec = vosk.KaldiRecognizer(self.model, self.sample_rate)
            self.rec.SetWords(True)
            return True
        except Exception as e:
            print(f"[STT] Ошибка инициализации распознавателя: {e}")
            return False
    
    def audio_callback(self, in_data, frame_count, time_info, status):
        if self.is_listening:
            self.audio_queue.put(in_data)
        return (None, pyaudio.paContinue)
    
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
                frames_per_buffer=self.chunk_size,
                stream_callback=self.audio_callback
            )
            
            stream.start_stream()
            
            print(f"[STT] Начало прослушивания")
            
            while self.is_listening:
                try:
                    audio_data = self.audio_queue.get(timeout=0.1)
                    
                    if self.rec.AcceptWaveform(audio_data):
                        result = json.loads(self.rec.Result())
                        text = result.get('text', '')
                        
                        if text:
                            print(f"[STT] Распознано: {text}")
                            self.current_text = text
                            
                            self.event_bus.emit("stt_text_ready", {
                                "text": text,
                                "is_final": True,
                                "timestamp": datetime.now().isoformat()
                            })
                            
                            self.event_bus.emit("llm_voice_input", {
                                "text": text,
                                "source": "microphone",
                                "timestamp": datetime.now().isoformat()
                            })
                    else:
                        partial = json.loads(self.rec.PartialResult())
                        partial_text = partial.get('partial', '')
                        
                        if partial_text:
                            self.event_bus.emit("stt_partial", {
                                "text": partial_text,
                                "is_final": False,
                                "timestamp": datetime.now().isoformat()
                            })
                    
                except queue.Empty:
                    pass
            
        except Exception as e:
            print(f"[STT] Ошибка воркера: {e}")
        finally:
            if stream:
                stream.stop_stream()
                stream.close()
            if p:
                p.terminate()
            print(f"[STT] Прослушивание остановлено")
    
    def start_listening(self):
        if self.is_listening:
            return
        
        if not self.init_recognizer():
            print(f"[STT] Не удалось инициализировать распознаватель")
            return
        
        self.is_listening = True
        self.audio_queue = queue.Queue()
        self.listen_thread = threading.Thread(target=self.listen_worker, daemon=True)
        self.listen_thread.start()
        print(f"[STT] Запущено прослушивание")
    
    def stop_listening(self):
        self.is_listening = False
        if self.listen_thread:
            self.listen_thread.join(timeout=2)
        print(f"[STT] Остановлено прослушивание")
    
    def register_routes(self):
        @self.app.route('/api/stt/status', methods=['GET'])
        def stt_status():
            return jsonify({
                "model_loaded": self.model_loaded,
                "is_listening": self.is_listening,
                "current_text": self.current_text,
                "available_devices": self.available_devices,
                "current_device": self.device_index,
                "model_name": self.model_name if self.model_loaded else None
            })
        
        @self.app.route('/api/stt/get_settings', methods=['GET'])
        def stt_get_settings():
            return jsonify({
                "device_index": self.device_index,
                "available_devices": self.available_devices,
                "model_loaded": self.model_loaded,
                "is_listening": self.is_listening
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
                
                return jsonify({
                    "status": "ok",
                    "device_index": device_index,
                    "message": "Устройство изменено"
                })
            
            return jsonify({"error": "Неверный индекс"}), 400
        
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
    
    def register_main_tab(self):
        return ("STT Голосовой ввод", '''
        <div class="row">
            <div class="col-md-12">
                <div class="card mb-3">
                    <div class="card-header">
                        <i class="fas fa-microphone me-2"></i>
                        Голосовое управление
                    </div>
                    <div class="card-body text-center">
                        <div id="sttStatus" class="mb-3">
                            <span class="badge bg-secondary" id="statusBadge">Загрузка...</span>
                        </div>
                        
                        <div id="voiceText" class="alert alert-info mb-3" style="display: none;">
                            <i class="fas fa-comment-dots me-2"></i>
                            <span id="recognizedText"></span>
                        </div>
                        
                        <div class="mb-3">
                            <button class="btn btn-success btn-lg" onclick="startSTT()" id="startBtn">
                                <i class="fas fa-play me-2"></i>Запустить
                            </button>
                            <button class="btn btn-warning btn-lg ms-2" onclick="restartSTT()" id="restartBtn">
                                <i class="fas fa-sync me-2"></i>Перезапустить
                            </button>
                            <button class="btn btn-danger btn-lg ms-2" onclick="stopSTT()" id="stopBtn">
                                <i class="fas fa-stop me-2"></i>Остановить
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <script>
            let statusInterval = null;
            let currentText = "";
            
            async function updateSTTStatus() {
                try {
                    const response = await fetch("/api/stt/status");
                    const data = await response.json();
                    
                    const statusBadge = document.getElementById("statusBadge");
                    if (!data.model_loaded) {
                        statusBadge.className = "badge bg-danger";
                        statusBadge.innerHTML = "❌ Модель не загружена";
                    } else if (data.is_listening) {
                        statusBadge.className = "badge bg-success";
                        statusBadge.innerHTML = "🎤 Слушаю...";
                    } else {
                        statusBadge.className = "badge bg-secondary";
                        statusBadge.innerHTML = "⏹️ Остановлено";
                    }
                    
                    if (data.current_text && data.current_text !== currentText) {
                        currentText = data.current_text;
                        showRecognizedText(currentText);
                    }
                } catch(e) {
                    console.error("Ошибка статуса:", e);
                }
            }
            
            function showRecognizedText(text) {
                const voiceTextDiv = document.getElementById("voiceText");
                const textSpan = document.getElementById("recognizedText");
                textSpan.innerHTML = text;
                voiceTextDiv.style.display = "block";
                
                setTimeout(function() {
                    voiceTextDiv.style.display = "none";
                }, 3000);
            }
            
            async function startSTT() {
                const response = await fetch("/api/stt/start", {method: "POST"});
                const data = await response.json();
                if (data.status === "ok") {
                    showNotification("Распознавание запущено", "success");
                    updateSTTStatus();
                }
            }
            
            async function stopSTT() {
                const response = await fetch("/api/stt/stop", {method: "POST"});
                const data = await response.json();
                if (data.status === "ok") {
                    showNotification("Распознавание остановлено", "info");
                    updateSTTStatus();
                }
            }
            
            async function restartSTT() {
                const response = await fetch("/api/stt/restart", {method: "POST"});
                const data = await response.json();
                if (data.status === "ok") {
                    showNotification("Распознавание перезапущено", "success");
                    updateSTTStatus();
                }
            }
            
            function showNotification(message, type) {
                const notification = document.createElement("div");
                notification.className = "alert alert-" + type + " position-fixed top-0 end-0 m-3";
                notification.style.zIndex = "9999";
                notification.style.animation = "fadeIn 0.3s ease";
                notification.style.background = type === "success" ? "#10b981" : (type === "error" ? "#ef4444" : "#3b82f6");
                notification.style.color = "#ffffff";
                notification.style.padding = "10px 20px";
                notification.style.borderRadius = "10px";
                notification.innerHTML = message;
                document.body.appendChild(notification);
                setTimeout(function() {
                    notification.remove();
                }, 2000);
            }
            
            statusInterval = setInterval(updateSTTStatus, 500);
            updateSTTStatus();
            
            window.addEventListener("beforeunload", function() {
                if (statusInterval) clearInterval(statusInterval);
            });
        </script>
        ''')
    
    def register_settings_ui(self):
        devices_html = ""
        for dev in self.available_devices:
            selected = "selected" if dev['index'] == self.device_index else ""
            devices_html += '<option value="' + str(dev['index']) + '" ' + selected + '>' + dev['name'] + '</option>\n'
        
        return '''
        <div class="row">
            <div class="col-md-12">
                <div class="card mb-3">
                    <div class="card-header">Устройство ввода</div>
                    <div class="card-body">
                        <div class="mb-3">
                            <label class="form-label">Выберите микрофон или виртуальное устройство</label>
                            <select class="form-select" id="deviceSelect" onchange="changeDevice()">
                                ''' + (devices_html if devices_html else '<option value="">Нет устройств</option>') + '''
                            </select>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <script>
            async function changeDevice() {
                const deviceIndex = parseInt(document.getElementById("deviceSelect").value);
                
                const response = await fetch("/api/stt/set_device", {
                    method: "POST",
                    headers: {"Content-Type": "application/json"},
                    body: JSON.stringify({device_index: deviceIndex})
                });
                
                const data = await response.json();
                if (data.status === "ok") {
                    showNotification("Устройство изменено", "success");
                } else {
                    showNotification("Ошибка: " + data.error, "error");
                }
            }
            
            function showNotification(message, type) {
                const notification = document.createElement("div");
                notification.className = "alert alert-" + type + " position-fixed top-0 end-0 m-3";
                notification.style.zIndex = "9999";
                notification.style.animation = "fadeIn 0.3s ease";
                notification.style.background = type === "success" ? "#10b981" : "#ef4444";
                notification.style.color = "#ffffff";
                notification.style.padding = "10px 20px";
                notification.style.borderRadius = "10px";
                notification.innerHTML = message;
                document.body.appendChild(notification);
                setTimeout(function() {
                    notification.remove();
                }, 2000);
            }
        </script>
        '''
    
    def on_load(self):
        if not self.model_loaded:
            print(f"[{self.display_name}] ВНИМАНИЕ: Модель не загружена!")
            print(f"[{self.display_name}] Скачайте модель с https://alphacephei.com/vosk/models")
            print(f"[{self.display_name}] И распакуйте в {self.model_path}")
        self.event_bus.subscribe("llm_voice_input", self.handle_voice_input)
        print(f"[{self.display_name}] Загружен")
        if self.model_loaded:
            print(f"[{self.display_name}] Vosk модель активна")
        else:
            print(f"[{self.display_name}] Модель не загружена, проверьте путь")
    
    def handle_voice_input(self, data):
        text = data.get('text', '')
        print(f"[STT] Голосовая команда передана в LLM: {text}")