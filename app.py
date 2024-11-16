import os
import json
import threading
import uuid
from pathlib import Path
from flask import Flask, jsonify, request, render_template, session
from pytube import YouTube
import whisper
from pydub import AudioSegment
from datetime import datetime

app = Flask(__name__)
app.secret_key = os.urandom(24)

# Create a directory for temporary files
TEMP_DIR = Path("temp_files")
TEMP_DIR.mkdir(exist_ok=True)

# Global dictionary to store transcription tasks
transcription_tasks = {}

class TranscriptionStatus:
    def __init__(self):
        self.progress = "Initialized"
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
                "progress": self.progress,
                "transcription": self.transcription,
                "error": self.error,
                "finished": self.finished
            }

class TranscriptionWorker(threading.Thread):
    def __init__(self, url, model_size, task_id):
        super().__init__()
        self.url = url
        self.model_size = model_size
        self.task_id = task_id
        self.temp_dir = TEMP_DIR / task_id
        self.temp_dir.mkdir(exist_ok=True)
        self.status = TranscriptionStatus()
        transcription_tasks[self.task_id] = self.status

    def update_status(self, **kwargs):
        self.status.update(**kwargs)

    def cleanup_files(self):
        """Remove all temporary files for this task"""
        if self.temp_dir.exists():
            for file in self.temp_dir.glob("*"):
                try:
                    file.unlink()
                except Exception as e:
                    print(f"Error deleting {file}: {e}")
            try:
                self.temp_dir.rmdir()
            except Exception as e:
                print(f"Error removing directory {self.temp_dir}: {e}")

    def download_audio(self):
        """Download audio from YouTube video"""
        try:
            self.update_status(progress="Downloading audio...")
            yt = YouTube(self.url)
            
            # Get audio stream
            stream = yt.streams.filter(only_audio=True).first()
            if not stream:
                raise Exception("No audio stream found")

            # Download and convert to WAV
            output_path = self.temp_dir / "audio.wav"
            mp4_path = self.temp_dir / "temp_audio.mp4"
            
            stream.download(output_path=str(self.temp_dir), filename="temp_audio.mp4")
            
            audio = AudioSegment.from_file(str(mp4_path))
            audio.export(str(output_path), format="wav")
            mp4_path.unlink()  # Remove the temporary MP4 file
            
            return output_path
            
        except Exception as e:
            raise Exception(f"Download failed: {str(e)}")

    def split_audio(self, input_file, chunk_length_ms=30000):
        """Split audio file into smaller chunks"""
        self.update_status(progress="Splitting audio into chunks...")
        audio = AudioSegment.from_file(str(input_file))
        chunks = []
        
        for i in range(0, len(audio), chunk_length_ms):
            chunk = audio[i:i + chunk_length_ms]
            chunk_file = self.temp_dir / f"chunk_{i // chunk_length_ms}.wav"
            chunk.export(str(chunk_file), format="wav")
            chunks.append(chunk_file)
            
        return chunks

    def transcribe_audio(self, filename):
        """Transcribe audio using Whisper"""
        model = whisper.load_model(self.model_size)
        result = model.transcribe(str(filename))
        return result["text"]

    def run(self):
        try:
            # Download audio
            audio_path = self.download_audio()

            # Split into chunks and transcribe
            chunks = self.split_audio(audio_path)
            
            full_transcription = []
            total_chunks = len(chunks)
            
            for i, chunk in enumerate(chunks, 1):
                self.update_status(progress=f"Transcribing part {i} of {total_chunks}...")
                text = self.transcribe_audio(chunk)
                full_transcription.append(text)

            # Join transcriptions and cleanup
            final_transcription = " ".join(full_transcription)
            self.update_status(
                transcription=final_transcription,
                progress="Transcription completed!",
                finished=True
            )

        except Exception as e:
            self.update_status(
                error=f"Error during transcription: {str(e)}",
                finished=True
            )
        finally:
            self.cleanup_files()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/transcribe', methods=['POST'])
def transcribe():
    try:
        data = request.get_json()
        if not data or 'video_url' not in data:
            return jsonify({"error": "No video URL provided"}), 400

        video_url = data['video_url']
        model_size = data.get('model_size', 'base')
        
        # Validate model size
        if model_size not in ['tiny', 'base', 'small', 'medium', 'large']:
            return jsonify({"error": "Invalid model size"}), 400

        # Generate unique task ID
        task_id = str(uuid.uuid4())
        
        # Store task ID in session
        session['current_task_id'] = task_id

        # Start transcription
        worker = TranscriptionWorker(video_url, model_size, task_id)
        worker.start()

        return jsonify({
            "message": "Transcription started",
            "task_id": task_id
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/progress')
def progress():
    task_id = session.get('current_task_id')
    
    if not task_id or task_id not in transcription_tasks:
        return jsonify({"error": "No active transcription task"}), 404

    status = transcription_tasks[task_id].get_status()
    
    # If the task is finished, prepare to clean up
    if status["finished"]:
        if task_id in transcription_tasks:
            if status["error"] is None:
                response = jsonify(status)
                # Only remove the task if it was successful
                del transcription_tasks[task_id]
                session.pop('current_task_id', None)
                return response
            else:
                return jsonify(status), 500
    
    return jsonify(status)

@app.route('/cleanup', methods=['POST'])
def cleanup():
    task_id = session.get('current_task_id')
    if task_id and task_id in transcription_tasks:
        # Ensure any remaining files are cleaned up
        temp_dir = TEMP_DIR / task_id
        if temp_dir.exists():
            for file in temp_dir.glob("*"):
                try:
                    file.unlink()
                except Exception:
                    pass
            try:
                temp_dir.rmdir()
            except Exception:
                pass
        
        del transcription_tasks[task_id]
        session.pop('current_task_id', None)
    
    return jsonify({"message": "Cleanup successful"})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8000))
    app.run(debug=True, port=port)
