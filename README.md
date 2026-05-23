# Automatic Speech Recognition using QuartzNet

This project provides an Automatic Speech Recognition (ASR) system trained on the LibriSpeech (English) dataset. It leverages fine-tuned QuartzNet and CitriNet models to convert audio speech into text, exposed via a fast, RESTful API endpoint for easy integration.

## Tech Stack
- Python 3.8+
- PyTorch
- NVIDIA NeMo (QuartzNet15x5, CitriNet)
- ONNX Runtime
- Flask

## How to Run
1. Install system prerequisites (e.g., `sox`):
   ```bash
   sudo apt-get install sox
   ```
2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Start the Flask API server:
   ```bash
   cd app
   python api.py
   ```

## API Usage
You can test the transcription endpoint by sending a POST request containing a `.wav` file to the `/audio` endpoint.

```bash
curl -X POST http://localhost:5000/audio \
  -F "audio=@../data/test_audio/sample.wav"
```
