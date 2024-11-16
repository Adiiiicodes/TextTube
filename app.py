import os
import json
import threading
from flask import Flask, jsonify, request, render_template, session
from pytube import YouTube
import whisper
from pydub import AudioSegment
from datetime import datetime

app = Flask(__name__)
app.secret_key = os.urandom(24)

# Global dictionary to store transcription tasks
transcription_tasks = {}

class TranscriptionStatus:
    def __init__(self):
        self.initialized = True
        self.progress = None
        self.transcription = None
        self.error = None
        self.finished = False
        self._lock = threading.Lock()

    def update(self, **kwargs):
        with self._lock:
            for key, value in kwargs.items():
                if hasattr(self, key):
                    setattr(self, key, value)

    def get_status(self):
        with self._lock:
            return {
                "initialized": self.initialized,
                "progress": self.progress,
                "transcription": self.transcription,
                "error": self.error,
                "finished": self.finished
            }

def load_cookies_from_env():
    """Load and validate cookies from environment variable"""
    try:
        cookies_json = os.environ.get("YOUTUBE_COOKIES")
        if not cookies_json:
            return []
        
        cookies = json.loads(cookies_json)
        # Validate cookie format
        required_fields = ['domain', 'name', 'value']
        for cookie in cookies:
            if not all(field in cookie for field in required_fields):
                raise ValueError("Invalid cookie format")
            
            # Convert expiration date if exists
            if 'expirationDate' in cookie:
                # Ensure it's not expired
                if float(cookie['expirationDate']) < datetime.now().timestamp():
                    continue
        
        return cookies
    except json.JSONDecodeError:
        print("Error decoding cookies JSON")
        return []
    except Exception as e:
        print(f"Error loading cookies: {str(e)}")
        return []

def validate_cookies():
    """Validate that necessary cookies are present and not expired"""
    cookies = load_cookies_from_env()
    required_cookies = ['__Secure-3PSID', 'LOGIN_INFO']
    
    found_cookies = set(cookie['name'] for cookie in cookies)
    return all(cookie in found_cookies for cookie in required_cookies)

class TranscriptionWorker(threading.Thread):
    def __init__(self, url, model_size, task_id):
        super().__init__()
        self.url = url
        self.model_size = model_size
        self.task_id = task_id
        self.status = TranscriptionStatus()
        transcription_tasks[self.task_id] = self.status

    def update_status(self, **kwargs):
        self.status.update(**kwargs)

    def download_audio_with_pytube(self, url, output_file="downloaded_audio.wav"):
        try:
            yt = YouTube(url)
            stream = yt.streams.filter(only_audio=True).first()
            # Download the audio
            stream.download(filename="downloaded_audio.mp4")

            # Convert the downloaded audio to WAV using pydub
            audio = AudioSegment.from_file("downloaded_audio.mp4")
            audio.export(output_file, format="wav")
            os.remove("downloaded_audio.mp4")  # Clean up the original download file
        except Exception as e:
            raise Exception(f"Download failed: {str(e)}")

    def split_audio(self, input_file, chunk_length_ms=30000):
        audio = AudioSegment.from_file(input_file)
        chunks = []
        for i in range(0, len(audio), chunk_length_ms):
            chunk = audio[i:i + chunk_length_ms]
            chunk_file = f"chunk_{i // chunk_length_ms}.wav"
            chunk.export(chunk_file, format="wav")
            chunks.append(chunk_file)
        return chunks

    def transcribe_audio_whisper(self, filename):
        model = whisper.load_model(self.model_size)
        result = model.transcribe(filename)
        return result["text"]

    def run(self):
        try:
            download_path = f"downloaded_audio_{self.task_id}.wav"

            self.update_status(progress="Starting download...")
            try:
                self.download_audio_with_pytube(self.url, download_path)
            except Exception as download_error:
                self.update_status(
                    error=f"Error during download: {str(download_error)}",
                    finished=True
                )
                return

            self.update_status(progress="Processing audio...")
            audio_chunks = self.split_audio(download_path)
            
            full_transcription = ""
            for i, chunk in enumerate(audio_chunks, 1):
                self.update_status(
                    progress=f"Transcribing part {i} of {len(audio_chunks)}..."
                )
                transcription = self.transcribe_audio_whisper(chunk)
                full_transcription += transcription + "\n"
                os.remove(chunk)

            self.update_status(
                transcription=full_transcription,
                progress="Transcription completed successfully!",
                finished=True
            )

            if os.path.exists(download_path):
                os.remove(download_path)

        except Exception as e:
            self.update_status(
                error=f"Unexpected error: {str(e)}",
                finished=True
            )

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/transcribe', methods=['POST'])
def transcribe():
    try:
        video_url = request.json.get('video_url')
        model_size = request.json.get('model_size', 'base')

        if not video_url:
            return jsonify({"error": "No video URL provided"}), 400

        if not validate_cookies():
            return jsonify({"error": "Invalid or missing YouTube cookies"}), 401

        # Generate a unique task ID
        task_id = str(hash(video_url + str(os.urandom(8))))
        
        # Store task ID in session
        session['current_task_id'] = task_id

        # Start the transcription worker
        worker = TranscriptionWorker(video_url, model_size, task_id)
        worker.start()

        return jsonify({
            "message": "Transcription started",
            "video_url": video_url,
            "task_id": task_id
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/progress')
def progress():
    task_id = session.get('current_task_id')
    
    if not task_id or task_id not in transcription_tasks:
        return jsonify({"error": "No active transcription task"}), 400

    status = transcription_tasks[task_id].get_status()

    if status["error"]:
        return jsonify({
            "progress": status["progress"],
            "error": status["error"],
            "finished": status["finished"],
        }), 500

    return jsonify({
        "progress": status["progress"],
        "transcription": status["transcription"],
        "error": None,
        "finished": status["finished"],
    })

@app.route('/cleanup', methods=['POST'])
def cleanup():
    task_id = session.get('current_task_id')
    if task_id and task_id in transcription_tasks:
        del transcription_tasks[task_id]
        session.pop('current_task_id', None)
    return jsonify({"message": "Cleanup successful"})

if __name__ == '__main__':
    # Check if cookies are properly configured
    if not validate_cookies():
        print("Warning: YouTube cookies are not properly configured!")
        print("Please set the YOUTUBE_COOKIES environment variable with valid cookies.")
    
    port = int(os.environ.get("PORT", 8000))
    app.run()
