document.getElementById('transcribeButton').addEventListener('click', function() {
    const url = document.getElementById('url').value;
    const model = document.querySelector('input[name="model"]:checked').value;

    if (!url) {
        alert('Please enter a YouTube URL');
        return;
    }

    // Update progress UI
    document.getElementById('progressBar').value = 0;
    document.getElementById('progressText').textContent = 'Starting transcription...';
    document.getElementById('transcriptionOutput').value = '';

    // Send transcription request
    fetch('/transcribe', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
            video_url: url,  // Changed from 'url' to 'video_url' to match backend
            model_size: model
        })
    })
    .then(response => response.json())
    .then(data => {
        console.log('Transcription started:', data);
        startProgressChecking();
    })
    .catch(error => {
        console.error('Error:', error);
        document.getElementById('progressText').textContent = 'Error starting transcription';
    });
});

function startProgressChecking() {
    const progressInterval = setInterval(() => {
        fetch('/progress')
        .then(response => response.json())
        .then(data => {
            console.log('Progress update:', data);  // Debug log

            // Update progress text
            if (data.progress) {
                document.getElementById('progressText').textContent = data.progress;
            }

            // Update transcription if available
            if (data.transcription) {
                document.getElementById('transcriptionOutput').value = data.transcription;
            }

            // Check for completion
            if (data.progress && data.progress.includes('completed')) {
                clearInterval(progressInterval);
            }

            // Check for errors
            if (data.error) {
                document.getElementById('progressText').textContent = `Error: ${data.error}`;
                clearInterval(progressInterval);
            }
        })
        .catch(error => {
            console.error('Progress check error:', error);
            document.getElementById('progressText').textContent = 'Error checking progress';
            clearInterval(progressInterval);
        });
    }, 1000);
}