from flask import Flask, request, jsonify
import requests
import subprocess
import tempfile
import torch

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
        return {
            'sample_rate': sample_rate,
            'duration': duration
        }
    except Exception as e:
        return jsonify({"error": e})

def transcribe_audio(files):
    with tempfile.TemporaryDirectory() as tmpdir:
        with open(os.path.join(tmpdir, 'manifest.json'), 'w') as fp:
            for audio_file in files:
                entry = {'audio_filepath': audio_file, 'duration': 100000, 'text': 'nothing'}
                fp.write(json.dumps(entry) + '\n')

        config = {'paths2audio_files': files, 'batch_size': 4, 'temp_dir': tmpdir}
        all_text = []
        temporary_datalayer = setup_transcribe_dataloader(config, quartznet.decoder.vocabulary)
        for test_batch in temporary_datalayer:
            #print(test_batch[0])
            processed_signal, processed_signal_len = quartznet.preprocessor(
                input_signal=test_batch[0].to(quartznet.device), length=test_batch[1].to(quartznet.device)
            )
            #print(1)
            ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(processed_signal),}
            ologits = ort_session.run(None, ort_inputs)
            alogits = np.asarray(ologits)
            logits = torch.from_numpy(alogits[0])
            greedy_predictions = logits.argmax(dim=-1, keepdim=False)
            wer = WER(decoding=quartznet.decoding, use_cer=False)
            hypotheses, _ = wer.decoding.ctc_decoder_predictions_tensor(greedy_predictions)
            if isinstance(hypotheses, list):
                for text in hypotheses:
                    all_text.append(text)
            else:
                all_text.append(hypotheses)
        return all_text


@app.route('/audio', methods = ['POST'])
def extract_audio():
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file provided"})

    audio_file = request.files['audio']

    file_path = audio_file.filename

    # future = executor.submit(get_audio_info, file_path)
    
    # audio_info = future.result()
    audio_info = get_audio_info(file_path)

    text = transcribe_audio(file_path)

    response_data = {
        "duration": audio_info.get("duration", 0),
        "sample_rate": audio_info.get("sample_rate", 0),
        "text": " ".join(text)
    }

    if audio_info:
        return jsonify(response_data)
    else:
        return jsonify({"error": "Failed to retrieve audio information"})

if __name__ == '__main__':
    app.run(threaded=True, debug=True)

    #WSL