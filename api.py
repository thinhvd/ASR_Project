from flask import Flask, request, jsonify
import requests
import subprocess
import tempfile
import torch

import glob
import json
import os
import tempfile
import onnxruntime

import numpy as np
import nemo.collections.asr as nemo_asr
from nemo.collections.asr.data.audio_to_text import AudioToCharDataset
from nemo.collections.asr.metrics.wer import WER

from setup import setup_transcribe_dataloader, to_numpy

app = Flask(__name__)

ort_session = onnxruntime.InferenceSession('qn.onnx', providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider'])
quartznet = nemo_asr.models.EncDecCTCModel.from_pretrained(model_name="QuartzNet15x5Base-En")

def get_audio_info(path):
    try:
        result = subprocess.check_output(['sox', '--info', path], 
                                        stderr=subprocess.STDOUT, text=True)
                                        
        sample_rate = 0
        duration = 0

        lines = result.split('\n')
        for line in lines:
            if "Sample Rate" in line:
                sample_rate = int(line.split(':')[1].strip())
            if "Duration" in line:
                hours = int(line.split(':')[1].strip())
                minutes = int(line.split(':')[2].strip())
                seconds = float((line.split(':')[3].strip()).split(" ")[0])
                duration = hours * 3600 + minutes * 60 + seconds

        return duration
    except Exception as e:
        return {"error": e}

def transcribe_audio(files):
    with tempfile.TemporaryDirectory() as tmpdir:
        with open(os.path.join(tmpdir, 'manifest.json'), 'w') as fp:
            for audio_file in files:
                entry = {'audio_filepath': audio_file, 'duration': 100000, 'text': 'nothing'}
                fp.write(json.dumps(entry) + '\n')

        config = {'paths2audio_files': files, 'batch_size': 4, 'temp_dir': tmpdir}

        temporary_datalayer = setup_transcribe_dataloader(config, quartznet.decoder.vocabulary)
        for test_batch in temporary_datalayer:
            processed_signal, processed_signal_len = quartznet.preprocessor(
                input_signal=test_batch[0].to(quartznet.device), length=test_batch[1].to(quartznet.device)
            )
            ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(processed_signal),}
            ologits = ort_session.run(None, ort_inputs)
            alogits = np.asarray(ologits)
            logits = torch.from_numpy(alogits[0])
            greedy_predictions = logits.argmax(dim=-1, keepdim=False)
            wer = WER(decoding=quartznet.decoding, use_cer=False)
            hypotheses, _ = wer.decoding.ctc_decoder_predictions_tensor(greedy_predictions)
            
        return hypotheses


@app.route('/audio', methods = ['POST'])
def extract_audio():
    if 'audio' not in request.files:
        return {"error": "No audio file provided"}

    all_audio_files = request.files.getlist('audio')
    temp_audio_paths = []

    for audio_file in all_audio_files:
        temp_audio_path = os.path.join(tempfile.gettempdir(), audio_file.filename)
        audio_file.save(temp_audio_path)
        temp_audio_paths.append(temp_audio_path)

    durations = [get_audio_info(path) for path in temp_audio_paths]

    transcriptions = transcribe_audio(temp_audio_paths)

    # Prepare result data
    result_data = []
    for i in range(len(all_audio_files)):
        result_entry = {
            "file_name": all_audio_files[i].filename,
            "duration": durations[i],
            "text": transcriptions[i],
        }
        result_data.append(result_entry)

    for temp_audio_path in temp_audio_paths:
        os.remove(temp_audio_path)

    if result_data:
        return jsonify(result_data)
    else:
        return {"error": "Failed to retrieve audio information"}

if __name__ == '__main__':
    app.run(threaded=True, debug=True)

    #WSL