from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

#route to serve the frontend
@app.route('/')
def index():
    return render_template('index.html')

#API endpoint for submitting prompts
@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    prompt = data.get('prompt', '')
    toxic = data.get('toxic', False)

    # NEEDS TO BE REPLACED WITH OUR MODEL'S LOGIC
    prediction = f"Generated continuation for: '{prompt}' (Toxic: {toxic})"
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)
