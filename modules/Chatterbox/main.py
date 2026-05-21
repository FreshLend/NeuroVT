import threading
import queue
import torch
import numpy as np
import pyaudio
import time
import base64
import os
import glob
from flask import request, jsonify
from flask_socketio import join_room, emit
from modules.base_module import BaseModule
from datetime import datetime
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

class Chatterbox(BaseModule):
    name = "Chatterbox"
    display_name = "TTS (Chatterbox)"

    OUTPUT_MODES = {
        "speakers": "Только динамики",
        "virtual": "Только виртуальный микрофон",
        "both": "Динамики + виртуальный микрофон"
    }

    def __init__(self, app, event_bus, socketio):
        super().__init__(app, event_bus, socketio)
        self.model = None
        self.model_loaded = False
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.sample_rate = 24000

        self.message_queue = queue.Queue()
        self.is_playing = False
        self.speech_history = []
        self.total_processed = 0

        self.pyaudio_instance = None
        self.audio_stream_speakers = None
        self.audio_stream_virtual = None

        self.output_device_speakers = None
        self.output_device_virtual = None
        self.output_mode = "speakers"
        self.available_output_devices = []

        self.references_dir = os.path.join(self.module_dir, "references")
        os.makedirs(self.references_dir, exist_ok=True)

        self.available_voices = []
        self.current_voice = None

        self.exaggeration = 0.85
        self.cfg_weight = 0.0
        self.temperature = 0.85
        self.min_p = 0.05
        self.top_p = 1.0
        self.repetition_penalty = 2.0

        self.load_settings()
        self.scan_voices()
        if self.available_voices and len(self.available_voices) > 0:
            default_voice = self.available_voices[0]['name']
            if self.current_voice is None or not any(v['name'] == self.current_voice for v in self.available_voices):
                self.current_voice = default_voice
                self.save_settings()
        self.scan_output_devices()
        self.init_pyaudio()

    def init_pyaudio(self):
        try:
            self.pyaudio_instance = pyaudio.PyAudio()
            print(f"[TTS] PyAudio инициализирован")
        except Exception as e:
            print(f"[TTS] Ошибка инициализации PyAudio: {e}")

    def scan_output_devices(self):
        try:
            if not self.pyaudio_instance:
                self.pyaudio_instance = pyaudio.PyAudio()
            self.available_output_devices = []
            for i in range(self.pyaudio_instance.get_device_count()):
                device_info = self.pyaudio_instance.get_device_info_by_index(i)
                if device_info['maxOutputChannels'] > 0:
                    self.available_output_devices.append({
                        'index': i,
                        'name': device_info['name'],
                        'channels': int(device_info['maxOutputChannels']),
                        'sample_rate': int(device_info['defaultSampleRate'])
                    })
            print(f"[TTS] Найдено выходных устройств: {len(self.available_output_devices)}")
        except Exception as e:
            print(f"[TTS] Ошибка сканирования устройств вывода: {e}")

    def scan_voices(self):
        self.available_voices = []
        wav_files = glob.glob(os.path.join(self.references_dir, "*.wav"))
        for wav_path in wav_files:
            voice_name = os.path.splitext(os.path.basename(wav_path))[0]
            self.available_voices.append({"name": voice_name, "path": wav_path})
        if not self.available_voices:
            self.available_voices.append({"name": "default", "path": None})
        print(f"[TTS] Найдено голосов: {len(self.available_voices)}")

    def load_settings(self):
        settings = self.load_module_settings()
        self.output_mode = settings.get('output_mode', 'speakers')
        self.output_device_speakers = settings.get('output_device_speakers')
        self.output_device_virtual = settings.get('output_device_virtual')
        self.current_voice = settings.get('current_voice')
        self.exaggeration = settings.get('exaggeration', 0.85)
        self.cfg_weight = settings.get('cfg_weight', 0.0)
        self.temperature = settings.get('temperature', 0.85)
        self.min_p = settings.get('min_p', 0.05)
        self.top_p = settings.get('top_p', 1.0)
        self.repetition_penalty = settings.get('repetition_penalty', 2.0)

        if self.available_output_devices:
            if self.output_device_speakers is None:
                self.output_device_speakers = self.available_output_devices[0]['index']
            if self.output_device_virtual is None:
                self.output_device_virtual = self.available_output_devices[0]['index']
            self.save_settings()

    def save_settings(self):
        self.save_module_settings({
            'output_mode': self.output_mode,
            'output_device_speakers': self.output_device_speakers,
            'output_device_virtual': self.output_device_virtual,
            'current_voice': self.current_voice,
            'exaggeration': self.exaggeration,
            'cfg_weight': self.cfg_weight,
            'temperature': self.temperature,
            'min_p': self.min_p,
            'top_p': self.top_p,
            'repetition_penalty': self.repetition_penalty
        })

    def load_model(self):
        try:
            print("[TTS] Загрузка модели Chatterbox...")
            if not torch.cuda.is_available():
                self.model = ChatterboxMultilingualTTS.from_pretrained(device='cpu')
            else:
                self.model = ChatterboxMultilingualTTS.from_pretrained(device=self.device)
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
        voice_path = None
        if self.current_voice and self.current_voice != "default":
            for v in self.available_voices:
                if v['name'] == self.current_voice:
                    voice_path = v['path']
                    break
        for part in text_parts:
            try:
                gen_kwargs = {
                    "text": part,
                    "language_id": 'ru',
                    "exaggeration": self.exaggeration,
                    "cfg_weight": self.cfg_weight,
                    "temperature": self.temperature,
                    "min_p": self.min_p,
                    "top_p": self.top_p,
                    "repetition_penalty": self.repetition_penalty
                }
                if voice_path and os.path.exists(voice_path):
                    gen_kwargs["audio_prompt_path"] = voice_path
                
                wav_tensor = self.model.generate(**gen_kwargs)
                duration = wav_tensor.shape[-1] / self.sample_rate
                wav_bytes = self.tensor_to_wav(wav_tensor, self.sample_rate)
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
        if self.output_mode == "speakers":
            self._play_on_device(audio_bytes, self.output_device_speakers, "speakers")
        elif self.output_mode == "virtual":
            self._play_on_device(audio_bytes, self.output_device_virtual, "virtual")
        elif self.output_mode == "both":
            t1 = threading.Thread(target=self._play_on_device, args=(audio_bytes, self.output_device_speakers, "speakers"))
            t2 = threading.Thread(target=self._play_on_device, args=(audio_bytes, self.output_device_virtual, "virtual"))
            t1.start()
            t2.start()
            t1.join()
            t2.join()

    def _play_on_device(self, audio_bytes, device_index, device_name):
        try:
            stream = None
            if device_name == "speakers" and self.audio_stream_speakers is not None:
                stream = self.audio_stream_speakers
            elif device_name == "virtual" and self.audio_stream_virtual is not None:
                stream = self.audio_stream_virtual
            if stream is None:
                stream = self.pyaudio_instance.open(
                    format=pyaudio.paInt16,
                    channels=1,
                    rate=self.sample_rate,
                    output=True,
                    output_device_index=device_index if device_index is not None else None,
                    frames_per_buffer=1024
                )
                if device_name == "speakers":
                    self.audio_stream_speakers = stream
                else:
                    self.audio_stream_virtual = stream
            stream.write(audio_bytes)
        except Exception as e:
            print(f"[TTS] Ошибка воспроизведения на {device_name}: {e}")

    def tensor_to_wav(self, audio_tensor, sample_rate):
        try:
            if audio_tensor.dim() == 1:
                audio_tensor = audio_tensor.unsqueeze(0)
            audio_np = audio_tensor.cpu().numpy()
            if audio_np.ndim == 2 and audio_np.shape[0] == 1:
                audio_np = audio_np[0]
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
                    audio_base64 = base64.b64encode(audio_data).decode('utf-8')
                    self.socketio.emit('tts_audio', {
                        'audio': audio_base64,
                        'text': text,
                        'source': source,
                        'duration': duration
                    }, room='tts_clients')
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
        @self.app.route('/api/tts/chatterbox/status', methods=['GET'])
        def tts_status():
            return jsonify({
                "model_loaded": self.model_loaded,
                "current_voice": self.current_voice,
                "voices": {v['name']: v['name'] for v in self.available_voices},
                "is_playing": self.is_playing,
                "queue_size": self.message_queue.qsize(),
                "total_processed": self.total_processed,
                "history": self.speech_history[-10:],
                "device": str(self.device),
                "sample_rate": self.sample_rate,
                "output_mode": self.output_mode,
                "output_device_speakers": self.output_device_speakers,
                "output_device_virtual": self.output_device_virtual,
                "exaggeration": self.exaggeration,
                "cfg_weight": self.cfg_weight,
                "temperature": self.temperature,
                "min_p": self.min_p,
                "top_p": self.top_p,
                "repetition_penalty": self.repetition_penalty
            })

        @self.app.route('/api/tts/chatterbox/get_settings', methods=['GET'])
        def tts_get_settings():
            return jsonify({
                "voice": self.current_voice,
                "available_voices": self.available_voices,
                "sample_rate": self.sample_rate,
                "quality": f"{self.sample_rate//1000} кГц",
                "output_mode": self.output_mode,
                "output_device_speakers": self.output_device_speakers,
                "output_device_virtual": self.output_device_virtual,
                "available_output_devices": self.available_output_devices,
                "output_modes": self.OUTPUT_MODES,
                "exaggeration": self.exaggeration,
                "cfg_weight": self.cfg_weight,
                "temperature": self.temperature,
                "min_p": self.min_p,
                "top_p": self.top_p,
                "repetition_penalty": self.repetition_penalty
            })

        @self.app.route('/api/tts/chatterbox/set_output_mode', methods=['POST'])
        def tts_set_output_mode():
            data = request.json
            mode = data.get('mode')
            if mode in self.OUTPUT_MODES:
                self.output_mode = mode
                self.save_settings()
                return jsonify({"status": "ok", "mode": mode})
            return jsonify({"error": "Неверный режим"}), 400

        @self.app.route('/api/tts/chatterbox/set_output_device', methods=['POST'])
        def tts_set_output_device():
            data = request.json
            device_type = data.get('type')
            device_index = data.get('device_index')
            if device_type == 'speakers':
                self.output_device_speakers = device_index
            elif device_type == 'virtual':
                self.output_device_virtual = device_index
            else:
                return jsonify({"error": "Неверный тип"}), 400
            self.save_settings()
            if device_type == 'speakers' and self.audio_stream_speakers:
                try:
                    self.audio_stream_speakers.close()
                except:
                    pass
                self.audio_stream_speakers = None
            if device_type == 'virtual' and self.audio_stream_virtual:
                try:
                    self.audio_stream_virtual.close()
                except:
                    pass
                self.audio_stream_virtual = None
            return jsonify({"status": "ok"})

        @self.app.route('/api/tts/chatterbox/queue/clear', methods=['POST'])
        def tts_clear_queue():
            cleared = self.message_queue.qsize()
            while not self.message_queue.empty():
                try:
                    self.message_queue.get_nowait()
                    self.message_queue.task_done()
                except queue.Empty:
                    break
            return jsonify({"status": "ok", "cleared": cleared})

        @self.app.route('/api/tts/chatterbox/voice', methods=['POST'])
        def tts_change_voice():
            data = request.json
            new_voice = data.get('voice', '')
            if any(v['name'] == new_voice for v in self.available_voices):
                self.current_voice = new_voice
                self.save_settings()
                return jsonify({"status": "ok", "voice": new_voice})
            return jsonify({"error": "Голос не найден"}), 400

        @self.app.route('/api/tts/chatterbox/rescan_voices', methods=['POST'])
        def tts_rescan_voices():
            self.scan_voices()
            if self.current_voice and not any(v['name'] == self.current_voice for v in self.available_voices):
                self.current_voice = self.available_voices[0]['name'] if self.available_voices else None
                self.save_settings()
            return jsonify({"status": "ok", "voices": self.available_voices, "current": self.current_voice})

        @self.app.route('/api/tts/chatterbox/params', methods=['POST'])
        def tts_update_params():
            data = request.json
            if 'exaggeration' in data:
                self.exaggeration = float(data['exaggeration'])
            if 'cfg_weight' in data:
                self.cfg_weight = float(data['cfg_weight'])
            if 'temperature' in data:
                self.temperature = float(data['temperature'])
            if 'min_p' in data:
                self.min_p = float(data['min_p'])
            if 'top_p' in data:
                self.top_p = float(data['top_p'])
            if 'repetition_penalty' in data:
                self.repetition_penalty = float(data['repetition_penalty'])
            self.save_settings()
            return jsonify({"status": "ok"})

        @self.app.route('/api/tts/chatterbox/test', methods=['POST'])
        def tts_test():
            data = request.json
            text = data.get('text', 'Привет! Это тестовая озвучка.')
            self.message_queue.put({"text": text, "source": "test", "timestamp": datetime.now().isoformat()})
            return jsonify({"status": "queued", "queue_position": self.message_queue.qsize()})

    def register_socketio_handlers(self, sio):
        @sio.on('join_tts')
        def handle_join_tts(data):
            join_room('tts_clients')
            print(f"[TTS] Клиент {request.sid} подключился к TTS")
            emit('tts_joined', {'status': 'ok'})

    def on_load(self):
        self.event_bus.subscribe("tts_speak", lambda data: self.message_queue.put(data))
        def load():
            if self.load_model():
                worker = threading.Thread(target=self.speak_worker, daemon=True)
                worker.start()
                print(f"[{self.display_name}] Готов")
            else:
                print(f"[{self.display_name}] Ошибка загрузки модели")
