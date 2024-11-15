from flask import Flask, jsonify, request, render_template
import threading
import os
import yt_dlp
import whisper
from pydub import AudioSegment
import time

app = Flask(__name__)

# Global worker instance
worker = None

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
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': 'downloaded_audio.%(ext)s',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
            }],
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
            chunk_progress = 50
            progress_per_chunk = 40 / len(audio_chunks)

            for i, chunk in enumerate(audio_chunks, 1):
                self.signals.progress = f"Transcribing part {i} of {len(audio_chunks)}..."
                transcription = self.transcribe_audio_whisper(chunk)
                full_transcription += transcription + "\n"
                os.remove(chunk)
                chunk_progress += progress_per_chunk

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
    try:
        print("Received transcribe request")  # Debug log
        global worker
        video_url = request.json.get('video_url')
        print(f"Video URL received: {video_url}")  # Debug log
        model_size = request.json.get('model_size', 'base')

        if not video_url:
            return jsonify({"error": "No video URL provided"}), 400

        # Prepare signal handler for worker
        signals = WorkerSignals()

        # Start the transcription worker in a background thread
        worker = TranscriptionWorker(video_url, model_size, signals)
        worker.start()

        return jsonify({"message": "Transcription started", "video_url": video_url}), 200
    except Exception as e:
        print(f"Error in transcribe route: {str(e)}")  # Debug log
        return jsonify({"error": str(e)}), 500

@app.route('/progress')
def progress():
    try:
        global worker
        if worker and worker.signals:
            return jsonify({
                "progress": worker.signals.progress,
                "transcription": worker.signals.transcription,
                "error": worker.signals.error
            })
        else:
            return jsonify({"error": "Worker not initialized"}), 500
    except Exception as e:
        print(f"Error in progress route: {str(e)}")  # Debug log
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=8000)