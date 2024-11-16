import os
import json
import threading
from flask import Flask, jsonify, request, render_template, session
import yt_dlp
import whisper
from pydub import AudioSegment

app = Flask(__name__)
app.secret_key = os.urandom(24)  # Add secret key for session management

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
    try:
        cookies_json = os.environ.get("YOUTUBE_COOKIES", "[]")
        return json.loads(cookies_json)
    except json.JSONDecodeError:
        return []

def validate_cookie_from_env():
    cookies = load_cookies_from_env()
    for cookie in cookies:
        if cookie.get("name") == "LOGIN_INFO":
            return True
    return False

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

    def download_audio_with_ytdlp(self, url, output_file="downloaded_audio.wav"):
        cookies = load_cookies_from_env()
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': 'downloaded_audio.%(ext)s',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav'
            }],
            'cookie_list': cookies,
            'quiet': True,
        }
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
            if os.path.exists("downloaded_audio.wav"):
                os.rename("downloaded_audio.wav", output_file)
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
                self.download_audio_with_ytdlp(self.url, download_path)
            except Exception as download_error:
                self.update_status(
                    error=f"Error during download: {download_error}",
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

        if not validate_cookie_from_env():
            return jsonify({"error": "Invalid or missing cookies"}), 401

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

# Cleanup route for completed tasks
@app.route('/cleanup', methods=['POST'])
def cleanup():
    task_id = session.get('current_task_id')
    if task_id and task_id in transcription_tasks:
        del transcription_tasks[task_id]
        session.pop('current_task_id', None)
    return jsonify({"message": "Cleanup successful"})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8000))
    app.run(debug=True, port=port)
