from flask import Flask, render_template, request, jsonify
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from CNNModel import CNNModel

app = Flask(__name__)

# Load model
model = CNNModel()
model.load_state_dict(torch.load("model/mnist_cnn.pth", map_location='cpu'))
model.eval()

# Preprocessing
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    image = Image.open(file).convert('L')
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(image)
        pred = F.softmax(output, dim=1)
        label = torch.argmax(pred, dim=1).item()

    return jsonify({'prediction': int(label)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
