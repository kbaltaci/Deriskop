from flask import Flask, request, jsonify, render_template
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)

# Modeli yükle
model = tf.keras.models.load_model('skin_cancer_model.h5')

# Görseli işlemek ve modelin tahmin yapabilmesi için uygun formata dönüştürmek
def prepare_image(img):
    img = img.resize((128, 128))  # Resmi boyutlandırma
    img_array = np.array(img)  # NumPy dizisine dönüştürme
    img_array = img_array / 255.0  # Ölçeklendirme
    return np.expand_dims(img_array, axis=0)  # Modelin tahmin yapabilmesi için boyutlandırma

@app.route('/')
def home():
    return render_template('index.html')  # HTML formunu render et

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"})
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"})
    
    if file:
        img = image.load_img(file, target_size=(128, 128))
        img_array = prepare_image(img)
        
        # Modeli kullanarak tahmin yap
        prediction = model.predict(img_array)
        
        # Sonuçları döndür
        result = 'Malignant' if prediction[0] > 0.5 else 'Benign'
        confidence = float(prediction[0]) * 100
        
        return render_template('index.html', prediction=result, confidence=confidence)

if __name__ == '__main__':
    app.run(debug=True)
