import logging
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
import cv2
import numpy as np
from typing import Optional

from src.config import Config, PROJECT_ROOT

logger = logging.getLogger(__name__)

class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True

class MJPEGRequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/stream.mjpg':
            self.send_response(200)
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
            self.end_headers()
            try:
                while True:
                    frame = self.server.streamer.get_jpeg_frame()
                    if frame is not None:
                        self.wfile.write(b'--frame\r\n')
                        self.send_header('Content-Type', 'image/jpeg')
                        self.send_header('Content-Length', str(len(frame)))
                        self.end_headers()
                        self.wfile.write(frame)
                        self.wfile.write(b'\r\n')
                    else:
                        time.sleep(0.1)
            except Exception as e:
                logger.debug(f"Stream client disconnected: {e}")
        else:
            self.send_response(404)
            self.end_headers()
            
    def log_message(self, format, *args):
        # Suppress default HTTP logging to avoid spam
        pass

class NetworkStreamer:
    """
    Network Streaming Module.
    Implements a simple MJPEG HTTP stream for local viewing.
    """
    def __init__(self, config: Config):
        self.config = config
        self.enabled = config.stream.enabled if hasattr(config, 'stream') and hasattr(config.stream, 'enabled') else True
        self.host = config.stream.host if hasattr(config, 'stream') and hasattr(config.stream, 'host') else '0.0.0.0'
        self.port = config.stream.port if hasattr(config, 'stream') and hasattr(config.stream, 'port') else 8080
        
        self._is_streaming = False
        self._current_frame_jpeg = None
        self._lock = threading.Lock()
        self._server = None
        self._thread = None

    @property
    def is_streaming(self) -> bool:
        return self._is_streaming

    def start(self):
        if not self.enabled:
            logger.info("Streaming is disabled in config.")
            return
            
        if self._is_streaming:
            return
            
        try:
            self._server = ThreadingHTTPServer((self.host, self.port), MJPEGRequestHandler)
            self._server.streamer = self
            self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
            self._thread.start()
            self._is_streaming = True
            logger.info(f"MJPEG Stream started at {self.get_stream_url()}")
        except Exception as e:
            logger.error(f"Failed to start streaming server: {e}")
            self._is_streaming = False

    def update_frame(self, frame: np.ndarray):
        if not self._is_streaming:
            return
            
        # Compress to JPEG
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 70]
        result, encimg = cv2.imencode('.jpg', frame, encode_param)
        if result:
            with self._lock:
                self._current_frame_jpeg = encimg.tobytes()

    def get_jpeg_frame(self) -> Optional[bytes]:
        with self._lock:
            return self._current_frame_jpeg

    def stop(self):
        if not self._is_streaming:
            return
            
        self._is_streaming = False
        if self._server:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
            
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
            
        logger.info("Streaming stopped.")

    def get_stream_url(self) -> str:
        return f"http://{self.host}:{self.port}/stream.mjpg"
