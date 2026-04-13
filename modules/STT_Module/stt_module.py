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
    
    def __init__(self, app, event_bus):
        super().__init__(app, event_bus)
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
        print(f"[STT] Загружены настройки: модель={self.model_size}, хоткей={self.hotkey}, язык={self.recognition_language}")
    
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
            print(f"[STT] Найдено {len(self.available_devices)} устройств ввода")
            
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
            print("[STT] Горячая клавиша: микрофон выключен")
        else:
            self.start_listening()
            print("[STT] Горячая клавиша: микрофон включён")
    
    def listen_worker(self):
        p = None
        stream = None
        
        try:
            p = pyaudio.PyAudio()
            
            print(f"[STT] Используется устройство ввода: {self.device_index}")
            
            for i in range(p.get_device_count()):
                device_info = p.get_device_info_by_index(i)
                if device_info['maxInputChannels'] > 0:
                    print(f"[STT] Устройство {i}: {device_info['name']}")
            
            stream = p.open(
                format=self.format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                input_device_index=self.device_index,
                frames_per_buffer=self.chunk_size
            )
            
            print("[STT] Поток открыт, началось прослушивание...")
            
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
                            print(f"[STT] Речь началась")
                    else:
                        if self.is_speaking and (time.time() - last_speech_time) > 0.8:
                            self.is_speaking = False
                            print(f"[STT] Речь закончилась")
                            
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
            print(f"[STT] Прослушивание остановлено")
    
    def process_audio(self, audio_bytes):
        try:
            audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            
            language_param = self.recognition_language if self.recognition_language else None
            
            segments, info = self.model.transcribe(
                audio_np,
                beam_size=5,
                language=language_param,
                vad_filter=True,
                vad_parameters={
                    "min_silence_duration_ms": 500,
                    "speech_pad_ms": 400
                }
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
        print(f"[STT] Запущено прослушивание")
    
    def stop_listening(self):
        if not self.is_listening:
            return
        
        self.is_listening = False
        
        if self.listen_thread:
            self.listen_thread.join(timeout=2)
        
        self.is_speaking = False
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
            print(f"[STT] Язык распознавания изменён на: {self.recognition_language or 'авто'}")
            
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
    
    def register_main_tab(self):
        return ("STT Голосовой ввод", f"""
        <div class="row" id="stt-module-container">
            <div class="col-md-12">
                <div class="card mb-3">
                    <div class="card-header">
                        <i class="fas fa-microphone me-2"></i>
                        Голосовое управление
                    </div>
                    <div class="card-body text-center">
                        <div id="sttStatus" class="mb-3">
                            <span class="badge bg-secondary" id="sttStatusBadge">Загрузка...</span>
                            <span class="badge bg-info ms-2" id="sttModelBadge">-</span>
                        </div>
                        
                        <div id="sttVoiceText" class="alert alert-info mb-3" style="display: none;">
                            <i class="fas fa-comment-dots me-2"></i>
                            <span id="sttRecognizedText"></span>
                        </div>
                        
                        <div class="mb-3">
                            <div id="sttSpeakingIndicator" style="display: none;">
                                <i class="fas fa-circle text-danger me-1" style="font-size: 12px;"></i>
                                <span>🎤 Говорите...</span>
                            </div>
                        </div>
                        
                        <div class="mb-3">
                            <button class="btn btn-success btn-lg" onclick="sttStart()" id="sttStartBtn">
                                <i class="fas fa-play me-2"></i>Запустить
                            </button>
                            <button class="btn btn-warning btn-lg ms-2" onclick="sttRestart()" id="sttRestartBtn">
                                <i class="fas fa-sync me-2"></i>Перезапустить
                            </button>
                            <button class="btn btn-danger btn-lg ms-2" onclick="sttStop()" id="sttStopBtn">
                                <i class="fas fa-stop me-2"></i>Остановить
                            </button>
                        </div>
                        
                        <div class="mt-3">
                            <div class="card">
                                <div class="card-header small">
                                    <i class="fas fa-history me-1"></i>
                                    История распознанных фраз
                                </div>
                                <div class="card-body" id="sttHistoryList" style="max-height: 300px; overflow-y: auto;">
                                    <div class="text-muted text-center">Нет записей</div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <script>
            let sttStatusInterval = null;
            let sttLastHistoryHash = '';
            let sttLastRecognizedText = '';
            let sttNotificationTimeout = null;
            
            function sttUpdateHistoryDisplay(history) {{
                const historyList = document.getElementById('sttHistoryList');
                
                const sttOnlyHistory = history ? history.filter(item => item.type === 'stt' || !item.type) : [];
                const newHash = JSON.stringify(sttOnlyHistory);
                
                if (newHash === sttLastHistoryHash) return;
                sttLastHistoryHash = newHash;
                
                if (!sttOnlyHistory || sttOnlyHistory.length === 0) {{
                    historyList.innerHTML = '<div class="text-muted text-center">Нет записей</div>';
                    return;
                }}
                
                const scrollPos = historyList.scrollTop;
                const isScrolledToBottom = historyList.scrollHeight - historyList.clientHeight <= scrollPos + 10;
                
                historyList.innerHTML = [...sttOnlyHistory].reverse().map(item => `
                    <div class="mb-2 p-2 border rounded">
                        <div class="d-flex justify-content-between mb-1">
                            <small class="text-muted">${{new Date(item.timestamp).toLocaleTimeString()}}</small>
                            <small class="badge bg-secondary">${{item.language || 'auto'}}</small>
                        </div>
                        <div class="mt-1">${{sttEscapeHtml(item.text)}}</div>
                    </div>
                `).join('');
                
                if (isScrolledToBottom) {{
                    historyList.scrollTop = historyList.scrollHeight;
                }}
            }}
            
            function sttShowRecognizedText(text) {{
                if (text === sttLastRecognizedText) return;
                sttLastRecognizedText = text;
                
                const voiceTextDiv = document.getElementById('sttVoiceText');
                const textSpan = document.getElementById('sttRecognizedText');
                
                textSpan.innerHTML = text;
                voiceTextDiv.style.display = 'block';
                
                if (sttNotificationTimeout) {{
                    clearTimeout(sttNotificationTimeout);
                }}
                
                sttNotificationTimeout = setTimeout(() => {{
                    voiceTextDiv.style.display = 'none';
                    sttNotificationTimeout = null;
                }}, 3000);
            }}
            
            async function sttUpdateStatus() {{
                try {{
                    const response = await fetch("/api/stt/status");
                    const data = await response.json();
                    
                    const statusBadge = document.getElementById("sttStatusBadge");
                    if (!data.model_loaded) {{
                        statusBadge.className = "badge bg-danger";
                        statusBadge.innerHTML = "❌ Модель не загружена";
                    }} else if (data.is_listening) {{
                        if (data.is_speaking) {{
                            statusBadge.className = "badge bg-danger";
                            statusBadge.innerHTML = "🎤 Слушаю... (речь)";
                            document.getElementById("sttSpeakingIndicator").style.display = "block";
                        }} else {{
                            statusBadge.className = "badge bg-success";
                            statusBadge.innerHTML = "🎤 Слушаю... (ожидание)";
                            document.getElementById("sttSpeakingIndicator").style.display = "none";
                        }}
                    }} else {{
                        statusBadge.className = "badge bg-secondary";
                        statusBadge.innerHTML = "⏹️ Остановлено";
                        document.getElementById("sttSpeakingIndicator").style.display = "none";
                    }}
                    
                    document.getElementById("sttModelBadge").innerHTML = data.model_size;
                    
                    if (data.current_text && data.current_text !== sttLastRecognizedText) {{
                        sttShowRecognizedText(data.current_text);
                    }}
                    
                    sttUpdateHistoryDisplay(data.history);
                    
                }} catch(e) {{
                    console.error("Ошибка статуса STT:", e);
                }}
            }}
            
            async function sttStart() {{
                const response = await fetch("/api/stt/start", {{method: "POST"}});
                const data = await response.json();
                if (data.status === "ok") {{
                    sttShowNotification("Распознавание запущено", "success");
                }}
            }}
            
            async function sttStop() {{
                const response = await fetch("/api/stt/stop", {{method: "POST"}});
                const data = await response.json();
                if (data.status === "ok") {{
                    sttShowNotification("Распознавание остановлено", "info");
                    sttLastRecognizedText = '';
                    document.getElementById('sttVoiceText').style.display = 'none';
                    if (sttNotificationTimeout) {{
                        clearTimeout(sttNotificationTimeout);
                        sttNotificationTimeout = null;
                    }}
                }}
            }}
            
            async function sttRestart() {{
                const response = await fetch("/api/stt/restart", {{method: "POST"}});
                const data = await response.json();
                if (data.status === "ok") {{
                    sttShowNotification("Распознавание перезапущено", "success");
                }}
            }}
            
            function sttShowNotification(message, type) {{
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
                setTimeout(() => notification.remove(), 2000);
            }}
            
            function sttEscapeHtml(text) {{
                const div = document.createElement('div');
                div.textContent = text;
                return div.innerHTML;
            }}
            
            window.sttStart = sttStart;
            window.sttStop = sttStop;
            window.sttRestart = sttRestart;
            
            sttStatusInterval = setInterval(sttUpdateStatus, 500);
            sttUpdateStatus();
            
            window.addEventListener("beforeunload", () => {{
                if (sttStatusInterval) clearInterval(sttStatusInterval);
                if (sttNotificationTimeout) clearTimeout(sttNotificationTimeout);
            }});
        </script>
        """)
    
    def register_settings_ui(self):
        devices_html = ""
        for dev in self.available_devices:
            selected = "selected" if dev['index'] == self.device_index else ""
            devices_html += f'<option value="{dev["index"]}" {selected}>{dev["name"]}</option>\n'
        
        languages = [
            {"code": None, "name": "🌐 Авто (определяется автоматически)"},
            {"code": "ru", "name": "🇷🇺 Русский"},
            {"code": "en", "name": "🇺🇸 English"},
            {"code": "fr", "name": "🇫🇷 Français"},
            {"code": "de", "name": "🇩🇪 Deutsch"},
            {"code": "es", "name": "🇪🇸 Español"},
            {"code": "it", "name": "🇮🇹 Italiano"},
            {"code": "zh", "name": "🇨🇳 中文"},
            {"code": "ja", "name": "🇯🇵 日本語"},
            {"code": "ko", "name": "🇰🇷 한국어"}
        ]
        
        language_options = ""
        for lang in languages:
            selected = "selected" if self.recognition_language == lang["code"] else ""
            value = "" if lang["code"] is None else lang["code"]
            language_options += f'<option value="{value}" {selected}>{lang["name"]}</option>\n'
        
        return f"""
        <div class="row" id="stt-settings-container">
            <div class="col-md-6">
                <div class="card mb-3">
                    <div class="card-header">Устройство ввода</div>
                    <div class="card-body">
                        <div class="mb-3">
                            <label class="form-label">Выберите микрофон</label>
                            <select class="form-select" id="sttDeviceSelect" onchange="sttChangeDevice()">
                                {devices_html if devices_html else '<option value="">Нет устройств</option>'}
                            </select>
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="col-md-6">
                <div class="card mb-3">
                    <div class="card-header">Язык распознавания</div>
                    <div class="card-body">
                        <div class="mb-3">
                            <label class="form-label">Выберите язык для распознавания речи</label>
                            <select class="form-select" id="sttLanguageSelect" onchange="sttChangeLanguage()">
                                {language_options}
                            </select>
                            <small class="text-muted d-block mt-2">
                                🌐 "Авто" - модель сама определит язык (рекомендуется для многоязычных сценариев)
                            </small>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="row">
            <div class="col-md-12">
                <div class="card mb-3">
                    <div class="card-header">Горячая клавиша</div>
                    <div class="card-body">
                        <div class="mb-3">
                            <label class="form-label">Клавиша для включения/выключения микрофона</label>
                            <input type="text" class="form-control" id="sttHotkeyInput" value="{self.hotkey}" 
                                placeholder="ctrl+shift+m" onchange="sttChangeHotkey()">
                            <small class="text-muted d-block mt-2">
                                Примеры: ctrl+shift+m, f1, alt+space, ctrl+alt+m
                            </small>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="alert alert-info" id="sttSettingsInfo">
            <i class="fas fa-info-circle me-2"></i>
            <span id="sttSettingsText">Загрузка...</span>
        </div>
        
        <script>
            async function sttLoadSettings() {{
                try {{
                    const response = await fetch('/api/stt/get_settings');
                    const data = await response.json();
                    const statusText = data.is_listening ? '🎤 Микрофон включён' : '⏹️ Микрофон выключен';
                    const languageName = data.recognition_language ? data.recognition_language.toUpperCase() : 'Авто';
                    document.getElementById('sttSettingsText').innerHTML = `
                        Модель: ${{data.model_size}} | Язык: ${{languageName}} | Хоткей: ${{data.hotkey}} | ${{statusText}}
                        ${{data.model_loaded ? '✅' : '⏳'}}
                    `;
                }} catch(e) {{
                    console.error('Ошибка загрузки настроек STT:', e);
                }}
            }}
            
            async function sttChangeDevice() {{
                const deviceIndex = parseInt(document.getElementById("sttDeviceSelect").value);
                
                const response = await fetch("/api/stt/set_device", {{
                    method: "POST",
                    headers: {{"Content-Type": "application/json"}},
                    body: JSON.stringify({{device_index: deviceIndex}})
                }});
                
                const data = await response.json();
                if (data.status === "ok") {{
                    sttShowNotification("Устройство изменено", "success");
                    sttLoadSettings();
                }}
            }}
            
            async function sttChangeLanguage() {{
                const select = document.getElementById("sttLanguageSelect");
                let language = select.value;
                
                if (language === "") {{
                    language = "auto";
                }}
                
                const response = await fetch("/api/stt/set_language", {{
                    method: "POST",
                    headers: {{"Content-Type": "application/json"}},
                    body: JSON.stringify({{language: language}})
                }});
                
                const data = await response.json();
                if (data.status === "ok") {{
                    const langDisplay = data.recognition_language ? data.recognition_language.toUpperCase() : 'Авто';
                    sttShowNotification("Язык распознавания изменён: " + langDisplay, "success");
                    sttLoadSettings();
                }} else {{
                    sttShowNotification("Ошибка: " + data.error, "error");
                }}
            }}
            
            async function sttChangeHotkey() {{
                const hotkey = document.getElementById("sttHotkeyInput").value;
                
                if (!hotkey) return;
                
                const response = await fetch("/api/stt/set_hotkey", {{
                    method: "POST",
                    headers: {{"Content-Type": "application/json"}},
                    body: JSON.stringify({{hotkey: hotkey}})
                }});
                
                const data = await response.json();
                if (data.status === "ok") {{
                    sttShowNotification("Горячая клавиша изменена: " + hotkey, "success");
                    sttLoadSettings();
                }} else {{
                    sttShowNotification("Ошибка: " + data.error, "error");
                }}
            }}
            
            function sttShowNotification(message, type) {{
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
                setTimeout(() => notification.remove(), 2000);
            }}
            
            window.sttChangeDevice = sttChangeDevice;
            window.sttChangeHotkey = sttChangeHotkey;
            window.sttChangeLanguage = sttChangeLanguage;
            
            sttLoadSettings();
        </script>
        """
    
    def on_load(self):
        self.event_bus.subscribe("llm_voice_input", self.handle_voice_input)
        print(f"[{self.display_name}] Загружен")
    
    def handle_voice_input(self, data):
        pass