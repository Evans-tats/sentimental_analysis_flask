from flask import Flask, request, jsonify
import numpy as np
from transformers import RobertaTokenizer
import onnxruntime as ort

app = Flask(__name__)

# Load tokenizer and ONNX model
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
session = ort.InferenceSession('roberta-sequence-classification-9.onnx')

@app.route('/predict', methods=['POST'])
def predict():
    # Parse input JSON
    text = request.json.get("text")
    if not text:
        return jsonify({"error": "No text provided"}), 400

    # Tokenize input text
    encoded = tokenizer(text, return_tensors='np', padding=True, truncation=True)
    input_ids = encoded['input_ids']
    attention_mask = encoded['attention_mask']

    # Prepare inputs for ONNX
    ort_inputs = {
        session.get_inputs()[0].name: input_ids,
        session.get_inputs()[1].name: attention_mask,
    }

    # Run inference
    ort_outs = session.run(None, ort_inputs)
    logits = ort_outs[0]
    prediction = int(np.argmax(logits, axis=1)[0])
    label = bool(prediction)

    return jsonify({'positive': label})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
