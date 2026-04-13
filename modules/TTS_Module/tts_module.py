import threading
import queue
import torch
import io
import wave
import numpy as np
import pyaudio
import time
from modules.base_module import BaseModule
from flask import jsonify, request
from datetime import datetime

class TTSModule(BaseModule):
    name = "tts"
    display_name = "TTS (Silero)"
    
    VOICES = {
        "aidar": "Айдар (мужской)",
        "baya": "Бая (женский)",
        "kseniya": "Ксения (женский)",
        "xenia": "Ксения (альт)",
        "eugene": "Евгений (мужской)"
    }
    
    def __init__(self, app, event_bus):
        super().__init__(app, event_bus)
        self.model = None
        self.model_loaded = False
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.sample_rate = 24000
        self.speaker = "xenia"
        
        self.message_queue = queue.Queue()
        self.is_playing = False
        self.speech_history = []
        self.total_processed = 0
        
        self.pyaudio_instance = None
        self.audio_stream = None
        
        self.load_settings()
        self.init_pyaudio()
    
    def init_pyaudio(self):
        try:
            self.pyaudio_instance = pyaudio.PyAudio()
            print(f"[TTS] PyAudio инициализирован")
        except Exception as e:
            print(f"[TTS] Ошибка инициализации PyAudio: {e}")
    
    def load_settings(self):
        settings = self.load_module_settings()
        
        saved_voice = settings.get('voice')
        if saved_voice and saved_voice in self.VOICES:
            self.speaker = saved_voice
        
        saved_rate = settings.get('sample_rate')
        if saved_rate in [24000, 48000]:
            self.sample_rate = saved_rate
    
    def save_settings(self):
        self.save_module_settings({
            'voice': self.speaker,
            'sample_rate': self.sample_rate
        })
    
    def load_model(self):
        try:
            print("[TTS] Загрузка модели...")
            
            torch.hub._validate_not_a_forked_repo = lambda a, b, c: True
            
            self.model, _ = torch.hub.load(
                repo_or_dir='snakers4/silero-models',
                model='silero_tts',
                language='ru',
                speaker='v3_1_ru',
                trust_repo=True,
                verbose=False
            )
            
            self.model.to(self.device)
            self.model_loaded = True
            
            print(f"[TTS] Модель загружена на {self.device}")
            return True
            
        except Exception as e:
            print(f"[TTS] Ошибка загрузки: {e}")
            return False

    def split_text_for_tts(self, text, max_length=400):
        if len(text) <= max_length:
            return [text]
        
        parts = []
        sentences = text.replace('!', '.').replace('?', '.').replace('\n', ' ').split('.')
        
        current_part = ""
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            if len(current_part) + len(sentence) + 2 <= max_length:
                if current_part:
                    current_part += ". " + sentence
                else:
                    current_part = sentence
            else:
                if current_part:
                    parts.append(current_part + ".")
                current_part = sentence
        
        if current_part:
            parts.append(current_part + ".")
        
        return parts if parts else [text[:max_length]]
    
    def text_to_speech(self, text):
        if not self.model_loaded or self.model is None:
            return None, 0
        
        text_parts = self.split_text_for_tts(text, 400)
        
        all_audio = []
        total_duration = 0
        
        for part in text_parts:
            try:
                if hasattr(self.model, 'apply_tts'):
                    audio = self.model.apply_tts(
                        text=part,
                        speaker=self.speaker,
                        sample_rate=self.sample_rate
                    )
                else:
                    audio = self.model(
                        text=part,
                        speaker=self.speaker,
                        sample_rate=self.sample_rate
                    )
                
                duration = len(audio) / self.sample_rate
                wav_bytes = self.tensor_to_wav(audio, self.sample_rate)
                if wav_bytes:
                    all_audio.append(wav_bytes)
                    total_duration += duration
                
                if len(text_parts) > 1:
                    time.sleep(0.1)
                    
            except Exception as e:
                print(f"[TTS] Ошибка генерации части: {e}")
                continue
        
        if all_audio:
            combined = b''.join(all_audio)
            return combined, total_duration
        
        return None, 0
    
    def play_audio(self, audio_bytes):
        try:
            if self.audio_stream is None:
                self.audio_stream = self.pyaudio_instance.open(
                    format=pyaudio.paInt16,
                    channels=1,
                    rate=self.sample_rate,
                    output=True,
                    frames_per_buffer=1024
                )
            
            self.audio_stream.write(audio_bytes)
            
        except Exception as e:
            print(f"[TTS] Ошибка воспроизведения: {e}")
    
    def tensor_to_wav(self, audio_tensor, sample_rate):
        try:
            if torch.is_tensor(audio_tensor):
                audio_np = audio_tensor.cpu().numpy()
            else:
                audio_np = np.array(audio_tensor)
            
            max_val = np.max(np.abs(audio_np))
            if max_val > 0:
                audio_np = audio_np / max_val
            audio_int16 = (audio_np * 32767).astype(np.int16)
            
            return audio_int16.tobytes()
            
        except Exception as e:
            print(f"[TTS] Ошибка конвертации: {e}")
            return None
    
    def speak_worker(self):
        print("[TTS] Запущен воркер")
        
        while True:
            try:
                data = self.message_queue.get(timeout=0.5)
                self.is_playing = True
                
                text = data.get('text', '')
                source = data.get('source', 'unknown')
                queue_position = self.message_queue.qsize() + 1
                
                print(f"[TTS] Озвучка: {text[:50]}...")
                
                self.event_bus.emit("tts_started", {
                    "text": text,
                    "source": source,
                    "queue_position": queue_position
                })
                
                start_time = time.time()
                audio_data, duration = self.text_to_speech(text)
                generation_time = time.time() - start_time
                
                if audio_data:
                    self.play_audio(audio_data)
                    
                    self.speech_history.append({
                        "text": text,
                        "source": source,
                        "timestamp": datetime.now().isoformat(),
                        "status": "completed",
                        "generation_time": round(generation_time, 2),
                        "duration": round(duration, 2)
                    })
                    
                    self.total_processed += 1
                    print(f"[TTS] Готово за {generation_time:.2f}с")
                    
                    time.sleep(0.1)
                else:
                    self.speech_history.append({
                        "text": text,
                        "source": source,
                        "timestamp": datetime.now().isoformat(),
                        "status": "error"
                    })
                    print(f"[TTS] Ошибка")
                    time.sleep(0.5)
                
                if len(self.speech_history) > 50:
                    self.speech_history = self.speech_history[-50:]
                
                self.is_playing = False
                self.message_queue.task_done()
                
            except queue.Empty:
                pass
            except Exception as e:
                print(f"[TTS] Ошибка: {e}")
                self.is_playing = False
                time.sleep(0.1)
    
    def register_routes(self):
        @self.app.route('/api/tts/status', methods=['GET'])
        def tts_status():
            return jsonify({
                "model_loaded": self.model_loaded,
                "current_voice": self.speaker,
                "voices": self.VOICES,
                "is_playing": self.is_playing,
                "queue_size": self.message_queue.qsize(),
                "total_processed": self.total_processed,
                "history": self.speech_history[-10:],
                "device": str(self.device),
                "sample_rate": self.sample_rate
            })
        
        @self.app.route('/api/tts/get_settings', methods=['GET'])
        def tts_get_settings():
            return jsonify({
                "voice": self.speaker,
                "voice_name": self.VOICES.get(self.speaker, self.speaker),
                "sample_rate": self.sample_rate,
                "quality": f"{self.sample_rate//1000} кГц"
            })
        
        @self.app.route('/api/tts/queue/clear', methods=['POST'])
        def tts_clear_queue():
            cleared = self.message_queue.qsize()
            while not self.message_queue.empty():
                try:
                    self.message_queue.get_nowait()
                    self.message_queue.task_done()
                except queue.Empty:
                    break
            return jsonify({"status": "ok", "cleared": cleared})
        
        @self.app.route('/api/tts/voice', methods=['POST'])
        def tts_change_voice():
            data = request.json
            new_voice = data.get('voice', '')
            
            if new_voice in self.VOICES:
                self.speaker = new_voice
                self.save_settings()
                return jsonify({"status": "ok", "voice": new_voice, "voice_name": self.VOICES[new_voice]})
            
            return jsonify({"error": "Голос не найден"}), 400
        
        @self.app.route('/api/tts/quality', methods=['POST'])
        def tts_change_quality():
            data = request.json
            new_rate = data.get('sample_rate', 24000)
            
            if new_rate in [24000, 48000]:
                self.sample_rate = new_rate
                self.save_settings()
                return jsonify({"status": "ok", "sample_rate": new_rate})
            
            return jsonify({"error": "Неверная частота"}), 400
        
        @self.app.route('/api/tts/test', methods=['POST'])
        def tts_test():
            data = request.json
            text = data.get('text', 'Привет! Это тестовая озвучка.')
            self.message_queue.put({"text": text, "source": "test", "timestamp": datetime.now().isoformat()})
            return jsonify({"status": "queued", "queue_position": self.message_queue.qsize()})
    
    def register_main_tab(self):
        return ("TTS Статус", f"""
        <div class="row" id="tts-module-container">
            <div class="col-md-6">
                <div class="card mb-3">
                    <div class="card-header">Статус озвучки</div>
                    <div class="card-body">
                        <div class="mb-2">
                            <strong>Состояние:</strong>
                            <span id="ttsStatus" class="badge bg-warning ms-2">Загрузка...</span>
                        </div>
                        <div class="mb-2">
                            <strong>Очередь:</strong>
                            <span id="ttsQueueSize">0</span>
                            <button class="btn btn-sm btn-danger ms-2" onclick="ttsClearQueue()">Очистить</button>
                        </div>
                        <div class="mb-2"><strong>Всего озвучено:</strong> <span id="ttsTotalProcessed">0</span></div>
                        <div class="mb-2"><strong>Голос:</strong> <span id="ttsCurrentVoice">-</span></div>
                        <div class="mb-2"><strong>Качество:</strong> <span id="ttsQuality">-</span></div>
                        <button class="btn btn-primary" onclick="ttsTest()">Тест озвучки</button>
                    </div>
                </div>
            </div>
            
            <div class="col-md-6">
                <div class="card">
                    <div class="card-header">История</div>
                    <div class="card-body" id="ttsHistoryList" style="max-height: 400px; overflow-y: auto;">
                        <div class="text-muted text-center">Нет сообщений</div>
                    </div>
                </div>
            </div>
        </div>
        
        <script>
            let ttsUpdateInterval = null;
            
            async function ttsUpdateStatus() {{
                try {{
                    const response = await fetch('/api/tts/status');
                    const data = await response.json();
                    
                    const statusBadge = document.getElementById('ttsStatus');
                    if (!data.model_loaded) statusBadge.innerHTML = 'Загрузка модели...';
                    else if (data.is_playing) statusBadge.innerHTML = 'Озвучивание...';
                    else if (data.queue_size > 0) statusBadge.innerHTML = `В очереди: ${{data.queue_size}}`;
                    else statusBadge.innerHTML = 'Готов';
                    
                    document.getElementById('ttsQueueSize').innerHTML = data.queue_size;
                    document.getElementById('ttsTotalProcessed').innerHTML = data.total_processed;
                    document.getElementById('ttsCurrentVoice').innerHTML = data.current_voice;
                    document.getElementById('ttsQuality').innerHTML = (data.sample_rate / 1000) + ' кГц';
                    
                    const historyList = document.getElementById('ttsHistoryList');
                    if (data.history.length === 0) {{
                        historyList.innerHTML = '<div class="text-muted text-center">Нет сообщений</div>';
                    }} else {{
                        historyList.innerHTML = [...data.history].reverse().map(item => `
                            <div class="mb-2 p-2 border rounded">
                                <div class="d-flex justify-content-between">
                                    <small>${{new Date(item.timestamp).toLocaleTimeString()}}</small>
                                    <small class="text-${{item.status === 'completed' ? 'success' : 'danger'}}">
                                        ${{item.status === 'completed' ? 'Озвучено' : 'Ошибка'}}
                                    </small>
                                </div>
                                <div class="mt-1 small">${{ttsEscapeHtml(item.text.substring(0, 100))}}</div>
                            </div>
                        `).join('');
                    }}
                }} catch(e) {{
                    console.error(e);
                }}
            }}
            
            async function ttsClearQueue() {{
                if (confirm('Очистить очередь?')) {{
                    await fetch('/api/tts/queue/clear', {{method: 'POST'}});
                }}
            }}
            
            async function ttsTest() {{
                const text = prompt('Введите текст:', 'Привет! Это тестовая озвучка.');
                if (text) {{
                    await fetch('/api/tts/test', {{
                        method: 'POST',
                        headers: {{'Content-Type': 'application/json'}},
                        body: JSON.stringify({{text: text}})
                    }});
                }}
            }}
            
            function ttsEscapeHtml(text) {{
                const div = document.createElement('div');
                div.textContent = text;
                return div.innerHTML;
            }}
            
            window.ttsClearQueue = ttsClearQueue;
            window.ttsTest = ttsTest;
            
            ttsUpdateInterval = setInterval(ttsUpdateStatus, 1000);
            ttsUpdateStatus();
            
            window.addEventListener('beforeunload', () => {{
                if (ttsUpdateInterval) clearInterval(ttsUpdateInterval);
            }});
        </script>
        """)
    
    def register_settings_ui(self):
        voice_options = ""
        for voice_id, voice_name in self.VOICES.items():
            selected = "selected" if voice_id == self.speaker else ""
            voice_options += f'<option value="{voice_id}" {selected}>{voice_name}</option>\n'
        
        quality_selected_24 = "selected" if self.sample_rate == 24000 else ""
        quality_selected_48 = "selected" if self.sample_rate == 48000 else ""
        
        return f"""
        <div class="row" id="tts-settings-container">
            <div class="col-md-6">
                <div class="mb-3">
                    <label class="form-label">Голос (автосохранение)</label>
                    <select class="form-select" id="ttsVoiceSelect" onchange="ttsChangeVoice()">
                        {voice_options}
                    </select>
                </div>
            </div>
            
            <div class="col-md-6">
                <div class="mb-3">
                    <label class="form-label">Качество (автосохранение)</label>
                    <select class="form-select" id="ttsQualitySelect" onchange="ttsChangeQuality()">
                        <option value="24000" {quality_selected_24}>Стандартное (24 кГц)</option>
                        <option value="48000" {quality_selected_48}>Высокое (48 кГц)</option>
                    </select>
                </div>
            </div>
        </div>
        
        <div class="alert alert-info" id="ttsCurrentSettings">Загрузка...</div>
        
        <script>
            async function ttsLoadSettings() {{
                const response = await fetch('/api/tts/get_settings');
                const data = await response.json();
                document.getElementById('ttsCurrentSettings').innerHTML = `
                    <i class="fas fa-info-circle me-2"></i>
                    Текущие: голос ${{data.voice_name}}, качество ${{data.quality}}
                `;
            }}
            
            async function ttsChangeVoice() {{
                const voice = document.getElementById('ttsVoiceSelect').value;
                const response = await fetch('/api/tts/voice', {{
                    method: 'POST',
                    headers: {{'Content-Type': 'application/json'}},
                    body: JSON.stringify({{voice: voice}})
                }});
                const data = await response.json();
                if (data.status === 'ok') {{
                    ttsShowNotification('Голос изменён', 'success');
                    ttsLoadSettings();
                }}
            }}
            
            async function ttsChangeQuality() {{
                const quality = parseInt(document.getElementById('ttsQualitySelect').value);
                const response = await fetch('/api/tts/quality', {{
                    method: 'POST',
                    headers: {{'Content-Type': 'application/json'}},
                    body: JSON.stringify({{sample_rate: quality}})
                }});
                const data = await response.json();
                if (data.status === 'ok') {{
                    ttsShowNotification('Качество изменено', 'success');
                    ttsLoadSettings();
                }}
            }}
            
            function ttsShowNotification(message, type) {{
                const notification = document.createElement('div');
                notification.className = `alert alert-${{type}} position-fixed top-0 end-0 m-3`;
                notification.style.zIndex = '9999';
                notification.style.background = type === 'success' ? '#10b981' : '#ef4444';
                notification.style.color = '#ffffff';
                notification.innerHTML = message;
                document.body.appendChild(notification);
                setTimeout(() => notification.remove(), 2000);
            }}
            
            window.ttsChangeVoice = ttsChangeVoice;
            window.ttsChangeQuality = ttsChangeQuality;
            
            ttsLoadSettings();
        </script>
        """
    
    def on_load(self):
        self.event_bus.subscribe("tts_speak", lambda data: self.message_queue.put(data))
        
        def load():
            if self.load_model():
                worker = threading.Thread(target=self.speak_worker, daemon=True)
                worker.start()
                print(f"[{self.display_name}] Готов")
            else:
                print(f"[{self.display_name}] Ошибка загрузки модели")
        
        threading.Thread(target=load, daemon=True).start()