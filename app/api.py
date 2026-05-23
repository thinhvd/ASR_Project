import os
import tempfile
import subprocess
import json
import logging
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import onnxruntime
import torch
import numpy as np
import nemo.collections.asr as nemo_asr
from nemo.collections.asr.metrics.wer import WER

from setup import setup_transcribe_dataloader, to_numpy
from config import Config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config.from_object(Config)

# Global variables for models
ort_session = None
quartznet = None

def load_models():
    """Initializes and loads the ONNX and NeMo models."""
    global ort_session, quartznet
    try:
        logger.info(f"Loading ONNX model from {app.config['ONNX_MODEL_PATH']}")
        # Fallback to CPU if GPU providers fail
        ort_session = onnxruntime.InferenceSession(
            app.config['ONNX_MODEL_PATH'],
            providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        
        logger.info(f"Loading NeMo preprocessor/vocabulary from {app.config['NEMO_MODEL_NAME']}")
        quartznet = nemo_asr.models.EncDecCTCModel.from_pretrained(model_name=app.config['NEMO_MODEL_NAME'])
        
        logger.info("Models loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load models: {str(e)}")
        raise e

# Load models at startup
with app.app_context():
    load_models()

def get_audio_info(path):
    """
    Extracts the duration of an audio file using sox.
    
    Args:
        path (str): The path to the audio file.
        
    Returns:
        float or dict: The duration in seconds if successful, else a dictionary with an error message.
    """
    try:
        result = subprocess.check_output(['sox', '--info', path], 
                                        stderr=subprocess.STDOUT, text=True)
                                        
        duration = 0

        lines = result.split('\n')
        for line in lines:
            if "Duration" in line:
                hours = int(line.split(':')[1].strip())
                minutes = int(line.split(':')[2].strip())
                seconds = float((line.split(':')[3].strip()).split(" ")[0])
                duration = hours * 3600 + minutes * 60 + seconds

        return duration
    except Exception as e:
        logger.error(f"Error getting audio info for {path}: {str(e)}")
        return {"error": str(e)}

def _transcribe_single(audio_file, duration, tmpdir):
    """Helper function to transcribe a single file."""
    manifest_path = os.path.join(tmpdir, f'manifest_{os.path.basename(audio_file)}.json')
    with open(manifest_path, 'w') as fp:
        entry = {'audio_filepath': audio_file, 'duration': duration, 'text': 'nothing'}
        fp.write(json.dumps(entry) + '\n')
        
    config = {'paths2audio_files': [audio_file], 'batch_size': 1, 'temp_dir': tmpdir}
    
    temporary_datalayer = setup_transcribe_dataloader(config, quartznet.decoder.vocabulary, manifest_path=manifest_path)
    
    for test_batch in temporary_datalayer:
        processed_signal, processed_signal_len = quartznet.preprocessor(
            input_signal=test_batch[0].to(quartznet.device), length=test_batch[1].to(quartznet.device)
        )
        ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(processed_signal)}
        ologits = ort_session.run(None, ort_inputs)
        alogits = np.asarray(ologits)
        logits = torch.from_numpy(alogits[0])
        greedy_predictions = logits.argmax(dim=-1, keepdim=False)
        wer = WER(decoding=quartznet.decoding, use_cer=False)
        hypotheses, _ = wer.decoding.ctc_decoder_predictions_tensor(greedy_predictions)
        return hypotheses[0]
        
    raise Exception("Empty dataloader or no transcript returned")

def transcribe_audio(file_paths, durations):
    """
    Transcribes a list of audio files using the loaded models.
    
    Args:
        file_paths (list): List of paths to audio files.
        durations (list): List of durations corresponding to each audio file.
        
    Returns:
        list: List of transcription strings or error messages.
    """
    transcriptions = []
    
    with tempfile.TemporaryDirectory() as tmpdir:
        manifest_path = os.path.join(tmpdir, 'manifest.json')
        with open(manifest_path, 'w') as fp:
            for audio_file, duration in zip(file_paths, durations):
                entry = {'audio_filepath': audio_file, 'duration': duration, 'text': 'nothing'}
                fp.write(json.dumps(entry) + '\n')

        config = {'paths2audio_files': file_paths, 'batch_size': app.config['BATCH_SIZE'], 'temp_dir': tmpdir}

        try:
            temporary_datalayer = setup_transcribe_dataloader(config, quartznet.decoder.vocabulary, manifest_path=manifest_path)
            
            for test_batch in temporary_datalayer:
                processed_signal, processed_signal_len = quartznet.preprocessor(
                    input_signal=test_batch[0].to(quartznet.device), length=test_batch[1].to(quartznet.device)
                )
                ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(processed_signal)}
                ologits = ort_session.run(None, ort_inputs)
                alogits = np.asarray(ologits)
                logits = torch.from_numpy(alogits[0])
                greedy_predictions = logits.argmax(dim=-1, keepdim=False)
                wer = WER(decoding=quartznet.decoding, use_cer=False)
                hypotheses, _ = wer.decoding.ctc_decoder_predictions_tensor(greedy_predictions)
                
                transcriptions.extend(hypotheses)
                
        except Exception as e:
            logger.warning(f"Batch transcription failed ({str(e)}). Falling back to sequential transcription to prevent full batch failure.")
            # Fallback to sequential transcription so one bad file doesn't crash the rest
            for audio_file, duration in zip(file_paths, durations):
                try:
                    transcription = _transcribe_single(audio_file, duration, tmpdir)
                    transcriptions.append(transcription)
                except Exception as single_e:
                    logger.error(f"Failed to transcribe {audio_file}: {str(single_e)}")
                    transcriptions.append({"error": str(single_e)})
            
    return transcriptions

@app.route('/audio', methods=['POST'])
def extract_audio():
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    all_audio_files = request.files.getlist('audio')
    if not all_audio_files or all_audio_files[0].filename == '':
        return jsonify({"error": "No audio file selected"}), 400

    temp_audio_paths = []
    valid_files_info = []
    result_data = []

    for audio_file in all_audio_files:
        filename = secure_filename(audio_file.filename)
        
        # Validate format
        if not filename.lower().endswith('.wav'):
            result_data.append({"file_name": filename, "error": "Invalid format. Only WAV is supported."})
            continue
            
        # Validate file size
        audio_file.seek(0, os.SEEK_END)
        file_size = audio_file.tell()
        audio_file.seek(0)
        
        if file_size > app.config['MAX_FILE_SIZE_BYTES']:
            result_data.append({"file_name": filename, "error": f"File exceeds maximum size of {app.config['MAX_FILE_SIZE_MB']}MB."})
            continue

        temp_audio_path = os.path.join(tempfile.gettempdir(), filename)
        try:
            audio_file.save(temp_audio_path)
            temp_audio_paths.append(temp_audio_path)
            valid_files_info.append({"file_name": filename, "path": temp_audio_path})
        except Exception as e:
            logger.error(f"Failed to save {filename}: {str(e)}")
            result_data.append({"file_name": filename, "error": "Failed to save file temporarily."})

    # If we have valid files, process them
    if valid_files_info:
        paths_to_process = []
        durations_to_process = []
        indices_to_process = []
        
        for idx, info in enumerate(valid_files_info):
            duration = get_audio_info(info['path'])
            if isinstance(duration, dict) and "error" in duration:
                result_data.append({
                    "file_name": info['file_name'],
                    "error": duration["error"]
                })
            else:
                paths_to_process.append(info['path'])
                durations_to_process.append(duration)
                indices_to_process.append(idx)
        
        if paths_to_process:
            try:
                transcriptions = transcribe_audio(paths_to_process, durations_to_process)
                
                for i, path_idx in enumerate(indices_to_process):
                    info = valid_files_info[path_idx]
                    trans_result = transcriptions[i] if i < len(transcriptions) else {"error": "Transcription missing"}
                    
                    if isinstance(trans_result, dict) and "error" in trans_result:
                        result_data.append({
                            "file_name": info['file_name'],
                            "error": trans_result["error"]
                        })
                    else:
                        result_data.append({
                            "file_name": info['file_name'],
                            "duration": durations_to_process[i],
                            "text": trans_result
                        })
            except Exception as e:
                logger.error(f"Critical error during batch transcription: {str(e)}")
                return jsonify({"error": "Internal server error during transcription."}), 500

    # Cleanup temporary files
    for temp_audio_path in temp_audio_paths:
        try:
            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)
        except Exception as e:
            logger.warning(f"Failed to remove temp file {temp_audio_path}: {str(e)}")

    if not result_data:
        return jsonify({"error": "No valid audio files processed"}), 400

    return jsonify(result_data), 200

if __name__ == '__main__':
    app.run(host=app.config['HOST'], port=app.config['PORT'], threaded=True, debug=False)