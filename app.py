from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import io

app = Flask(__name__)

# Modeli yükle
interpreter = tf.lite.Interpreter(model_path='models/skin_cancer_model.tflite')
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def predict_image(image):
    img = cv2.resize(image, (128, 128))
    img = img / 255.0
    img = np.expand_dims(img, axis=0).astype(np.float32)

    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    predictions = interpreter.get_tensor(output_details[0]['index'])
    return predictions[0][0]

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return "Dosya yok"
        file = request.files['file']
        if file.filename == '':
            return "Dosya seçilmedi"
        if file:
            image = Image.open(io.BytesIO(file.read()))
            image = np.array(image)
            prediction = predict_image(image)
            if prediction > 0.5:
                result = "Kötü Huylu (Malignant)"
            else:
                result = "İyi Huylu (Benign)"
            return render_template('index.html', prediction=result)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)