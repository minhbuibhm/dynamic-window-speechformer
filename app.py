import os
import requests
import base64
import time
import librosa
from flask import Flask, redirect, render_template, request, session, jsonify
from loguru import logger
import json
import torch
from datetime import datetime
# from extract_feature.extract_hubert import Hubert, extract_hubert
# from model.speech_dw_former import SpeechDW_Former
# from model.speechformer_v2 import SpeechFormer_v2

app = Flask(__name__)

save_feature_file = {
    "dw": "/content/drive/MyDrive/speechformer2/iemocap/feature/t_sne_dw.pickle",
    "former2": "/content/drive/MyDrive/speechformer2/iemocap/feature/t_sne_former2.pickle"
}
checkpoint_file = {
    "dw": "/content/drive/MyDrive/speechformer2/exp/speechdw/fold_1/best.pt",
    "former2": "/content/drive/MyDrive/speechformer2/exp/speechformer++/fold_1/best.pt"
}
model_config_file = "/content/drive/MyDrive/speechformer2/config/model_config.json"
feature_config_file = "/content/drive/MyDrive/speechformer2/config/iemocap_feature_config.json"
ckpt_path = "/content/drive/MyDrive/speechformer2/pre_trained_model/hubert/hubert_large_ll60k.pt"  # hubert_large_ll60k, hubert_base_ls960

sad_emoji = "😢"
happy_emoji = "😊"
neutral_emoji = "😐"
angry_emoji = "😡"

conveters = {"0": f"Giận dữ {angry_emoji}", "1": f"Bình thường {neutral_emoji}",
"2": f"Vui và phấn khích {happy_emoji}", "3": f"Buồn {sad_emoji}"}

def filter_checkpoint_weights(input_state_dict):
    state_dict = input_state_dict.copy()  # Create a copy to avoid modifying original
    filtered_state_dict = {}

    for key, value in state_dict.items():
        if key.startswith('module.'):  # Handle non-module prefixes (if present)
            key = key.split('module.')[1]
        filtered_state_dict[key] = value
    return filtered_state_dict

def load_dw_speech():
    model_type = "SpeechDW"
    dataset = "iemocap"
    feature = "hubert12"
    
    with open(model_config_file, 'r') as f1, open(feature_config_file, 'r') as f2:
        model_json = json.load(f1)[model_type]
        feas_json = json.load(f2)
        data_json = feas_json[feature]
        data_json['meta_csv_file'] = feas_json['meta_csv_file']
        model_json['num_classes'] = feas_json['num_classes']
        model_json['input_dim'] = (data_json['feature_dim'] // model_json['num_heads']) * model_json['num_heads']
        model_json['length'] = data_json['length']
        model_json['ffn_embed_dim'] = model_json['input_dim'] // 2
        model_json['hop'] = data_json['hop']

    extractor = SpeechDW_Former(**model_json).to('cuda')
    ckpt = torch.load(checkpoint_file["dw"])
    
    filtered_state_dict = filter_checkpoint_weights(ckpt['model'])
    extractor.load_state_dict(filtered_state_dict)
    
    return extractor

def load_speechformer2():
    model_type = "SpeechDW"
    dataset = "iemocap"
    feature = "hubert12"
    
    with open(model_config_file, 'r') as f1, open(feature_config_file, 'r') as f2:
        model_json = json.load(f1)[model_type]
        feas_json = json.load(f2)
        data_json = feas_json[feature]
        data_json['meta_csv_file'] = feas_json['meta_csv_file']
        model_json['num_classes'] = feas_json['num_classes']
        model_json['input_dim'] = (data_json['feature_dim'] // model_json['num_heads']) * model_json['num_heads']
        model_json['length'] = data_json['length']
        model_json['ffn_embed_dim'] = model_json['input_dim'] // 2
        model_json['hop'] = data_json['hop']

    extractor = SpeechFormer_v2(**model_json).to('cuda')
    ckpt = torch.load(checkpoint_file["former2"])
    
    filtered_state_dict = filter_checkpoint_weights(ckpt['model'])
    extractor.load_state_dict(filtered_state_dict)
    
    return extractor

# model1 = load_dw_speech()
# model2 = load_speechformer2()
# feature_extractor = Hubert(ckpt_path)

STATIC_DIR = "static"
UPLOAD_DIR = "upload"
RECORD_DIR = "record"

os.makedirs(os.path.join(STATIC_DIR, UPLOAD_DIR), exist_ok=True)
os.makedirs(os.path.join(STATIC_DIR, RECORD_DIR), exist_ok=True)

def run_app(use_ngrok=False):
    port = 5001
    if not use_ngrok:
        app.run(host="0.0.0.0", port=port, debug=False)
    
    from pyngrok  import ngrok
    ngrok.set_auth_token("21pUluBuD2HUj5Qr4Scfkcg6mrW_2bHd1pZ7jvr4Y1yJoV7nN")
    public_url=ngrok.connect(port).public_url
    print(f"public url: {public_url}")
    app.run(port=port)

@app.route("/")
def index():
    return render_template(
        template_name_or_list="index.html",
        audio_path=None,
    )

@app.route('/upload', methods=['POST', 'GET'])
def handle_upload():
    if request.method == "POST":
        _file = request.files['file']
        if _file.filename == '':
            return index()
        logger.info(f'file uploaded: {_file.filename}')
        filepath = os.path.join(STATIC_DIR, UPLOAD_DIR, _file.filename)
        _file.save(filepath)
        logger.info(f'saved file to: {filepath}')
        
        # hubert_feature = extract_hubert(feature_extractor, 12, filepath, "")
        
        # out = model1(hubert_feature.unsqueeze(0))
        # y_pred = torch.argmax(out, dim=1)
        # result1 = conveters[str(y_pred[0].item())]
        
        # out = model2(hubert_feature.unsqueeze(0))
        # y_pred = torch.argmax(out, dim=1)
        # result2 = conveters[str(y_pred[0].item())]

        result1 = "upload 1"
        result2 = "upload 2"
        
        return render_template(
            template_name_or_list='index.html',
            result1=result1,
            result2=result2,
            audiopath=filepath
        )
    else:
        return redirect("/")

@app.route('/submit_record', methods=['POST', 'GET'])
def handle_submit_record():
    if request.method == "POST":
        data = request.get_json()
        audio_base64 = data.get('audio')
        
        if not audio_base64:
            return jsonify({'error': 'No audio data provided'}), 400

        audio_data = base64.b64decode(audio_base64)
        filepath = os.path.join(STATIC_DIR, RECORD_DIR, f"{datetime.now().strftime('%d-%m/%Y-%H-%M-%S')}.wav")
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
        with open(filepath, 'wb') as f:
            f.write(audio_data)
        logger.info(f'saved file to: {filepath}')
        
        # hubert_feature = extract_hubert(feature_extractor, 12, filepath, "")
        # # logger.info(f"hubert feature: {hubert_feature}")
        # out = model1(hubert_feature.unsqueeze(0))
        # y_pred = torch.argmax(out, dim=1)
        # result1 = conveters[str(y_pred[0].item())]

        # out = model2(hubert_feature.unsqueeze(0))
        # y_pred = torch.argmax(out, dim=1)
        # result2 = conveters[str(y_pred[0].item())]
        
        result1 = "submit 1"
        result2 = "submit 2"
        
        return_data = {"emotion1": result1, "emotion2": result2, "filepath": filepath}
        print(return_data)
        return jsonify(return_data)
    else:
        return redirect("/")

if __name__ == '__main__':
    run_app(use_ngrok=True)
