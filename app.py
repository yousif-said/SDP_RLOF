from flask import Flask, render_template, request, jsonify
import json
import random
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

app = Flask(__name__)

# Load the dataset
with open('datasets/dataset.json', 'r') as f:
    dataset = json.load(f)

# Load GPT-2 model and tokenizer
model_name = 'gpt2'
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Route to serve the frontend
@app.route('/')
def index():
    return render_template('index.html')

# API endpoint for generating continuation
@app.route('/generate', methods=['POST'])
def generate():
    # Select a random prompt from the dataset
    prompt = random.choice(dataset)

    # Generate continuation using GPT-2
    inputs = tokenizer.encode(prompt, return_tensors='pt')
    outputs = model.generate(inputs, max_length=100, num_return_sequences=1)
    continuation = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Stop at the first period
    continuation = continuation.split('.')[0] + '.'

    return jsonify({'prompt': prompt, 'continuation': continuation})

# API endpoint for labeling the result
@app.route('/label', methods=['POST'])
def label():
    data = request.json
    prompt = data.get('prompt', '')
    continuation = data.get('continuation', '')
    toxic = data.get('toxic', False)

    # Store the result in a file
    result = {
        'prompt': prompt,
        'continuation': continuation,
        'toxic': toxic
    }
    with open('results.json', 'a') as f:
        json.dump(result, f)
        f.write('\n')

    return jsonify({'status': 'success'})

if __name__ == '__main__':
    app.run(debug=True)