from flask import Flask, request, render_template
import numpy as np
from tensorflow import lite
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

# TFLite modelini yükle
interpreter = lite.Interpreter(model_path="skin_cancer_model.tflite")
interpreter.allocate_tensors()

# Görseli işlemek ve modelin tahmin yapabilmesi için uygun formata dönüştürmek
def prepare_image(img):
    img = img.resize((128, 128))  # Resmi boyutlandırma
    img_array = np.array(img)  # NumPy dizisine dönüştürme
    img_array = img_array / 255.0  # Ölçeklendirme
    return np.expand_dims(img_array, axis=0)  # Modelin tahmin yapabilmesi için boyutlandırma

# TensorFlow Lite ile tahmin yapma
def predict(image_array):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Giriş verisini al
    input_data = np.array(image_array, dtype=np.float32)

    # Tahmin işlemi
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Çıktıyı al
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data[0][0]  # Modelin çıktısını döndür

@app.route('/')
def home():
    return render_template('index.html')  # HTML formunu render et

@app.route('/predict', methods=['POST'])
def predict_image():
    if 'file' not in request.files:
        return render_template('index.html', prediction="No file part")
    
    file = request.files['file']
    
    if file.filename == '':
        return render_template('index.html', prediction="No selected file")
    
    if file:
        img = image.load_img(file, target_size=(128, 128))
        img_array = prepare_image(img)
        
        # Modeli kullanarak tahmin yap
        prediction = predict(img_array)
        
        # Sonuçları döndür
        result = 'Malignant' if prediction > 0.5 else 'Benign'
        confidence = float(prediction) * 100
        
        return render_template('index.html', prediction=result, confidence=confidence)

if __name__ == '__main__':
    app.run(debug=True)
