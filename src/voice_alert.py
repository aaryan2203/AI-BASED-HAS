import logging
import queue
import threading
import time
from typing import Optional

from src.config import Config, PROJECT_ROOT

logger = logging.getLogger(__name__)

class VoiceAlertManager:
    """
    Offline Voice Alert System using pyttsx3.
    Runs in a dedicated background thread for COM safety on Windows.
    """
    def __init__(self, config: Config):
        self.config = config
        self.language = 'en'
        self.cooldown_seconds = config.voice.cooldown if hasattr(config, 'voice') and hasattr(config.voice, 'cooldown') else 3.0
        
        self._queue = queue.Queue()
        self._stop_event = threading.Event()
        self._worker_thread = threading.Thread(target=self._worker, daemon=True)
        self._last_alert_time = 0.0
        self._last_alert_text = ""
        
        self.enabled = config.voice.enabled if hasattr(config, 'voice') and hasattr(config.voice, 'enabled') else True
        
        if self.enabled:
            self._worker_thread.start()
        else:
            logger.info("Voice alerts are disabled in config.")

    def _worker(self):
        try:
            import pyttsx3
            engine = pyttsx3.init()
            
            # Simple voice selection based on language if possible, otherwise default
            voices = engine.getProperty('voices')
            # (In a full implementation, you might select specific voices for EN/HI)
            
            while not self._stop_event.is_set():
                try:
                    # Wait for next text
                    text = self._queue.get(timeout=0.1)
                    if text:
                        engine.say(text)
                        engine.runAndWait()
                except queue.Empty:
                    continue
                except Exception as e:
                    logger.error(f"Error in pyttsx3 engine: {e}")
        except ImportError:
            logger.warning("pyttsx3 is not installed. Voice alerts will be disabled.")
            self.enabled = False
        except Exception as e:
            logger.error(f"Failed to initialize pyttsx3: {e}")
            self.enabled = False

    def speak(self, text: str):
        if not self.enabled:
            return
            
        current_time = time.time()
        if text == self._last_alert_text and (current_time - self._last_alert_time) < self.cooldown_seconds:
            logger.debug(f"Skipping alert due to cooldown: {text}")
            return
            
        self._last_alert_text = text
        self._last_alert_time = current_time
        
        try:
            self._queue.put_nowait(text)
        except queue.Full:
            logger.warning("Voice alert queue is full, dropping message.")

    def set_language(self, lang: str):
        if lang in ['en', 'hi']:
            self.language = lang
        else:
            logger.warning(f"Unsupported language: {lang}. Keeping current language: {self.language}")

    def _get_template(self, key: str) -> str:
        # Fallback templates if config doesn't have them
        templates = {
            'en': {
                'step_completed': "Step completed. Next step: {next_step}",
                'warning_skip': "Warning: Step skipped. Expected {expected}",
                'warning_sequence': "Warning: Out of sequence. Expected {expected}",
                'experiment_complete': "Experiment completed successfully.",
                'astronaut_detected': "Astronaut detected. Ready to begin.",
                'low_confidence': "Low confidence detection."
            },
            'hi': {
                'step_completed': "कदम पूरा हुआ। अगला कदम: {next_step}",
                'warning_skip': "चेतावनी: कदम छोड़ दिया गया। अपेक्षित {expected}",
                'warning_sequence': "चेतावनी: क्रम से बाहर। अपेक्षित {expected}",
                'experiment_complete': "प्रयोग सफलतापूर्वक पूरा हुआ।",
                'astronaut_detected': "अंतरिक्ष यात्री का पता चला। शुरू करने के लिए तैयार।",
                'low_confidence': "कम विश्वास का पता लगाना।"
            }
        }
        
        try:
            return self.config.voice.languages[self.language][key]
        except (AttributeError, KeyError):
            return templates[self.language].get(key, templates['en'].get(key, ""))

    def announce_step_completed(self, next_step_name: str):
        template = self._get_template('step_completed')
        self.speak(template.format(next_step=next_step_name))

    def announce_warning_skip(self, expected_step: str):
        template = self._get_template('warning_skip')
        self.speak(template.format(expected=expected_step))

    def announce_warning_sequence(self, expected_step: str):
        template = self._get_template('warning_sequence')
        self.speak(template.format(expected=expected_step))

    def announce_experiment_complete(self):
        template = self._get_template('experiment_complete')
        self.speak(template)

    def announce_astronaut_detected(self):
        template = self._get_template('astronaut_detected')
        self.speak(template)

    def stop(self):
        if self.enabled:
            self._stop_event.set()
            if self._worker_thread.is_alive():
                self._worker_thread.join(timeout=2.0)
