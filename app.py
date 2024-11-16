import os
import json
import threading
from flask import Flask, jsonify, request, render_template
import yt_dlp
import whisper
from pydub import AudioSegment

app = Flask(__name__)

# Global variables
worker = None
worker_status = {
    "initialized": False,
    "progress": "No task started",
    "transcription": None,
    "error": None,
    "finished": False,
}


# Load cookies from environment
def load_cookies_from_env():
    try:
        cookies_json = os.environ.get("YOUTUBE_COOKIES", "[]")
        return json.loads(cookies_json)
    except json.JSONDecodeError:
        return []


# Validate cookies
def validate_cookie_from_env():
    cookies = load_cookies_from_env()
    for cookie in cookies:
        if cookie.get("name") == "LOGIN_INFO":  # Example validation
            return True
    return False


class WorkerSignals:
    """Signals to communicate between the worker and Flask app."""

    def __init__(self):
        self.progress = None
        self.transcription = None
        self.error = None
        self.finished = None


class TranscriptionWorker(threading.Thread):
    def __init__(self, url, model_size, signals):
        super().__init__()
        self.url = url
        self.model_size = model_size
        self.signals = signals

    def download_audio_with_ytdlp(self, url, output_file="downloaded_audio.wav"):
        cookies = load_cookies_from_env()
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': 'downloaded_audio.%(ext)s',
            'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'wav'}],
            'cookie_list': cookies,
            'quiet': True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        if os.path.exists("downloaded_audio.wav"):
            os.rename("downloaded_audio.wav", output_file)

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
            download_path = "downloaded_audio.wav"

            self.signals.progress = "Starting download..."
            self.download_audio_with_ytdlp(self.url, download_path)
            self.signals.progress = "Download completed!"

            self.signals.progress = "Processing audio..."
            audio_chunks = self.split_audio(download_path)
            self.signals.progress = "Audio processing completed!"

            full_transcription = ""
            for i, chunk in enumerate(audio_chunks, 1):
                self.signals.progress = f"Transcribing part {i} of {len(audio_chunks)}..."
                transcription = self.transcribe_audio_whisper(chunk)
                full_transcription += transcription + "\n"
                os.remove(chunk)

            self.signals.transcription = full_transcription
            self.signals.progress = "Transcription completed successfully!"

            if os.path.exists(download_path):
                os.remove(download_path)

            self.signals.finished = True

        except Exception as e:
            self.signals.error = str(e)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/transcribe', methods=['POST'])
def transcribe():
    global worker, worker_status
    try:
        video_url = request.json.get('video_url')
        model_size = request.json.get('model_size', 'base')

        if not video_url:
            return jsonify({"error": "No video URL provided"}), 400

        # Validate cookies before proceeding
        if not validate_cookie_from_env():
            return jsonify({"error": "Invalid or missing cookies"}), 401

        # Reset worker and status
        worker_status = {
            "initialized": True,
            "progress": "Task initialized...",
            "transcription": None,
            "error": None,
            "finished": False,
        }

        # Start the transcription worker
        signals = WorkerSignals()
        worker = TranscriptionWorker(video_url, model_size, signals)
        worker.start()

        return jsonify({"message": "Transcription started", "video_url": video_url}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/progress')
def progress():
    global worker, worker_status
    if not worker_status["initialized"]:
        return jsonify({"error": "No active transcription task"}), 400

    return jsonify({
        "progress": worker_status.get("progress"),
        "transcription": worker_status.get("transcription"),
        "error": worker_status.get("error"),
        "finished": worker_status.get("finished"),
    })


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8000))  # Use port from environment if provided
    app.run(debug=True, port=port)
