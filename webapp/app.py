from flask import Flask, request, jsonify
import torch
import numpy as np
from transformers import RobertaTokenizer
import onnxruntime

app = Flask(__name__)

tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
session = onnxruntime.InferenceSession('roberta-sequence-classification-9.onnx')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.json[0]
    input_ids = torch.tensor(tokenizer.encode(text, add_special_tokens=True)).unsqueeze(0)

    if input_ids.requires_grad:
        input_array = input_ids.detach().cpu().numpy().astype(np.int64)
    else:
        input_array = input_ids.cpu().numpy().astype(np.int64)

    inputs = {session.get_inputs()[0].name: input_array}
    out = session.run(None, inputs)

    results = np.argmax(out[0])  # Ensure you're indexing output properly

    return jsonify({'positive': bool(results)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
