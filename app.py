import os
import io
import base64
from flask import Flask, jsonify, request, Response, stream_with_context
from pytube import YouTube
import whisper
import torch
from pydub import AudioSegment
import numpy as np

app = Flask(__name__)

# Configure maximum content length for file uploads
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024  # 200MB max

def create_error_response(message, status_code=500):
    return jsonify({"error": message}), status_code

def download_audio(url):
    """Download audio from YouTube video and return as bytes"""
    try:
        yt = YouTube(url)
        stream = yt.streams.filter(only_audio=True).first()
        
        if not stream:
            raise Exception("No audio stream found")

        # Download to memory buffer
        buffer = io.BytesIO()
        stream.stream_to_buffer(buffer)
        buffer.seek(0)
        
        # Convert to WAV using pydub
        audio = AudioSegment.from_file(buffer, format="mp4")
        wav_buffer = io.BytesIO()
        audio.export(wav_buffer, format="wav")
        wav_buffer.seek(0)
        
        return wav_buffer.read()
        
    except Exception as e:
        raise Exception(f"Failed to download audio: {str(e)}")

def transcribe_audio_chunk(audio_data, model):
    """Transcribe a chunk of audio data"""
    try:
        # Convert audio data to the format whisper expects
        audio_segment = AudioSegment.from_wav(io.BytesIO(audio_data))
        audio_array = np.array(audio_segment.get_array_of_samples())
        
        # Normalize audio
        audio_float32 = audio_array.astype(np.float32) / 32768.0
        
        # Transcribe
        result = model.transcribe(audio_float32)
        return result["text"]
    except Exception as e:
        raise Exception(f"Transcription failed: {str(e)}")

def split_audio(audio_data, chunk_duration_ms=30000):
    """Split audio data into chunks"""
    audio = AudioSegment.from_wav(io.BytesIO(audio_data))
    chunks = []
    
    for i in range(0, len(audio), chunk_duration_ms):
        chunk = audio[i:i + chunk_duration_ms]
        chunk_buffer = io.BytesIO()
        chunk.export(chunk_buffer, format="wav")
        chunks.append(chunk_buffer.getvalue())
    
    return chunks

def stream_transcription(audio_data, model_name):
    """Generator function to stream transcription results"""
    try:
        # Load Whisper model
        model = whisper.load_model(model_name)
        
        # Split audio into chunks
        chunks = split_audio(audio_data)
        total_chunks = len(chunks)
        
        # Stream progress and results
        yield json.dumps({"status": "starting", "total_chunks": total_chunks}) + "\n"
        
        transcription = []
        for i, chunk in enumerate(chunks, 1):
            # Transcribe chunk
            text = transcribe_audio_chunk(chunk, model)
            transcription.append(text)
            
            # Stream progress
            progress = {
                "status": "processing",
                "chunk": i,
                "total_chunks": total_chunks,
                "partial_text": text
            }
            yield json.dumps(progress) + "\n"
        
        # Send final result
        final_result = {
            "status": "completed",
            "transcription": " ".join(transcription)
        }
        yield json.dumps(final_result) + "\n"
        
    except Exception as e:
        error_msg = {"status": "error", "error": str(e)}
        yield json.dumps(error_msg) + "\n"
    
    # Clear CUDA cache if using GPU
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

@app.route('/')
def index():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>YouTube Transcriber</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
            .status { margin: 20px 0; }
            #result { white-space: pre-wrap; }
        </style>
    </head>
    <body>
        <h1>YouTube Video Transcriber</h1>
        <div>
            <input type="text" id="videoUrl" placeholder="Enter YouTube URL" style="width: 300px;">
            <select id="modelSize">
                <option value="tiny">Tiny (Fastest)</option>
                <option value="base" selected>Base (Recommended)</option>
                <option value="small">Small</option>
                <option value="medium">Medium</option>
                <option value="large">Large (Best Quality)</option>
            </select>
            <button onclick="startTranscription()">Transcribe</button>
        </div>
        <div class="status" id="status"></div>
        <div id="result"></div>

        <script>
            function startTranscription() {
                const videoUrl = document.getElementById('videoUrl').value;
                const modelSize = document.getElementById('modelSize').value;
                const statusDiv = document.getElementById('status');
                const resultDiv = document.getElementById('result');
                
                statusDiv.textContent = 'Starting transcription...';
                resultDiv.textContent = '';
                
                fetch('/transcribe', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        video_url: videoUrl,
                        model_size: modelSize
                    })
                }).then(response => {
                    const reader = response.body.getReader();
                    const decoder = new TextDecoder();
                    
                    function processStream({ done, value }) {
                        if (done) return;
                        
                        const text = decoder.decode(value);
                        const lines = text.split('\\n').filter(line => line.trim());
                        
                        lines.forEach(line => {
                            try {
                                const data = JSON.parse(line);
                                
                                switch(data.status) {
                                    case 'starting':
                                        statusDiv.textContent = 'Preparing transcription...';
                                        break;
                                    case 'processing':
                                        statusDiv.textContent = `Transcribing chunk ${data.chunk} of ${data.total_chunks}...`;
                                        resultDiv.textContent += data.partial_text + ' ';
                                        break;
                                    case 'completed':
                                        statusDiv.textContent = 'Transcription completed!';
                                        resultDiv.textContent = data.transcription;
                                        break;
                                    case 'error':
                                        statusDiv.textContent = `Error: ${data.error}`;
                                        break;
                                }
                            } catch (e) {
                                console.error('Error parsing stream data:', e);
                            }
                        });
                        
                        return reader.read().then(processStream);
                    }
                    
                    return reader.read().then(processStream);
                }).catch(error => {
                    statusDiv.textContent = `Error: ${error.message}`;
                });
            }
        </script>
    </body>
    </html>
    """

@app.route('/transcribe', methods=['POST'])
def transcribe():
    try:
        data = request.get_json()
        if not data or 'video_url' not in data:
            return create_error_response("No video URL provided", 400)

        video_url = data['video_url']
        model_size = data.get('model_size', 'base')
        
        # Validate model size
        if model_size not in ['tiny', 'base', 'small', 'medium', 'large']:
            return create_error_response("Invalid model size", 400)

        # Download audio
        audio_data = download_audio(video_url)
        
        # Stream the transcription response
        return Response(
            stream_with_context(stream_transcription(audio_data, model_size)),
            mimetype='text/event-stream'
        )

    except Exception as e:
        return create_error_response(str(e))

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8000))
    app.run(debug=True, port=port)
