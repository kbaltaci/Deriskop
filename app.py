from flask import Flask, request, render_template, jsonify
import numpy as np
import tensorflow as tf
import os
app = Flask(__name__)

# TensorFlow Lite modelini yükle
interpreter = tf.lite.Interpreter(model_path="skin_cancer_model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Ana sayfa
@app.route('/')
def home():
    return render_template('index.html')

# Tahmin endpoint'i
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})

    # Görseli yükle ve ön işleme yap
    img = tf.keras.preprocessing.image.load_img(file, target_size=(128, 128))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Tahmin yap
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])
    result = "Malignant" if prediction[0][0] > 0.5 else "Benign"

    return jsonify({'prediction': result})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))
