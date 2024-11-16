// State management
let isTranscribing = false;
let progressInterval = null;

// UI Elements
const elements = {
    transcribeButton: document.getElementById('transcribeButton'),
    urlInput: document.getElementById('url'),
    progressBar: document.getElementById('progressBar'),
    progressText: document.getElementById('progressText'),
    transcriptionOutput: document.getElementById('transcriptionOutput')
};

// Utility functions
function updateUI(state) {
    elements.transcribeButton.disabled = state.isTranscribing;
    elements.urlInput.disabled = state.isTranscribing;
    elements.progressBar.value = state.progress || 0;
    elements.progressText.textContent = state.message || '';
    if (state.transcription) {
        elements.transcriptionOutput.value = state.transcription;
    }
}

function showError(message) {
    elements.progressText.textContent = `Error: ${message}`;
    elements.progressText.style.color = 'red';
    resetState();
}

function resetState() {
    isTranscribing = false;
    if (progressInterval) {
        clearInterval(progressInterval);
        progressInterval = null;
    }
    elements.transcribeButton.disabled = false;
    elements.urlInput.disabled = false;
}

function parseProgress(progressText) {
    // Extract percentage from progress text if possible
    if (progressText.includes('%')) {
        const match = progressText.match(/(\d+)%/);
        return match ? parseInt(match[1]) : 0;
    }
    // Estimate progress based on stage
    if (progressText.includes('Starting')) return 5;
    if (progressText.includes('download')) return 20;
    if (progressText.includes('Processing')) return 40;
    if (progressText.includes('Transcribing')) {
        const match = progressText.match(/part (\d+) of (\d+)/);
        if (match) {
            const [_, current, total] = match;
            return 40 + (parseInt(current) / parseInt(total) * 50);
        }
    }
    if (progressText.includes('completed')) return 100;
    return 0;
}

// Main transcription handler
async function handleTranscription(url, model) {
    try {
        isTranscribing = true;
        updateUI({
            isTranscribing: true,
            progress: 0,
            message: 'Starting transcription...'
        });

        const response = await fetch('/transcribe', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                video_url: url,
                model_size: model
            })
        });

        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.error || 'Failed to start transcription');
        }

        console.log('Transcription started:', data);
        startProgressChecking();

    } catch (error) {
        console.error('Transcription error:', error);
        showError(error.message || 'Failed to start transcription');
    }
}

// Progress checking
async function checkProgress() {
    try {
        const response = await fetch('/progress');
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.error || 'Failed to check progress');
        }

        console.log('Progress update:', data);

        // Update UI with progress
        if (data.progress) {
            const progressPercentage = parseProgress(data.progress);
            updateUI({
                isTranscribing: true,
                progress: progressPercentage,
                message: data.progress,
                transcription: data.transcription
            });
        }

        // Handle completion
        if (data.finished || (data.progress && data.progress.includes('completed'))) {
            updateUI({
                isTranscribing: false,
                progress: 100,
                message: 'Transcription completed!',
                transcription: data.transcription
            });
            resetState();
            
            // Clean up the task
            await fetch('/cleanup', { method: 'POST' });
        }

        // Handle errors
        if (data.error) {
            throw new Error(data.error);
        }

    } catch (error) {
        console.error('Progress check error:', error);
        showError(error.message || 'Error checking progress');
    }
}

function startProgressChecking() {
    if (progressInterval) {
        clearInterval(progressInterval);
    }
    progressInterval = setInterval(checkProgress, 1000);
}

// Event Listeners
elements.transcribeButton.addEventListener('click', function() {
    const url = elements.urlInput.value.trim();
    const model = document.querySelector('input[name="model"]:checked')?.value || 'base';

    if (!url) {
        showError('Please enter a YouTube URL');
        return;
    }

    if (!url.includes('youtube.com/') && !url.includes('youtu.be/')) {
        showError('Please enter a valid YouTube URL');
        return;
    }

    handleTranscription(url, model);
});

// Add keypress handler for URL input
elements.urlInput.addEventListener('keypress', function(e) {
    if (e.key === 'Enter' && !isTranscribing) {
        elements.transcribeButton.click();
    }
});

// Cleanup on page unload
window.addEventListener('beforeunload', async () => {
    if (isTranscribing) {
        try {
            await fetch('/cleanup', { method: 'POST' });
        } catch (error) {
            console.error('Cleanup error:', error);
        }
    }
});
