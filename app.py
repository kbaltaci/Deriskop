import gdown
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image
import numpy as np
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

# Flask uygulaması
app = Flask(__name__)

# Fotoğraf yükleme ayarları
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# uploads klasörünü kontrol et, yoksa oluştur
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Google Drive'dan modeli indirme
MODEL_PATH = "skin_cancer_model.h5"
GDRIVE_URL = 'https://drive.google.com/uc?export=download&id=1eiQgjDutm0ycugeiTEBSm1gkwylk1x_i'

# Modeli indir
if not os.path.exists(MODEL_PATH):
    print("Model dosyası bulunamadı, Google Drive'dan indiriliyor...")
    gdown.download(GDRIVE_URL, MODEL_PATH, quiet=False)
else:
    print("Model dosyası bulundu, yükleniyor...")

# Modeli yükle
model = keras.models.load_model(MODEL_PATH)

# İzin verilen dosya uzantıları
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Tahmin fonksiyonu
def predict_image(img_path):
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    prediction= prediction[0][0] * 100  # Yüzde olarak döndür
    if(prediction>90):
        prediction=prediction-50
    elif(prediction>80):
        prediction=prediction-50
    elif(prediction>70):
        prediction=prediction-40
    elif(prediction>60):
        prediction=prediction-40
    elif(prediction>50):
        prediction=prediction-40
    elif(prediction<50):
        prediction=prediction/5
    return prediction
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            file = request.files['file']
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)

                # Tahmin yap
                result_percentage = predict_image(filepath)
                return render_template('index.html', prediction=result_percentage, image_file=filename)
            else:
                return "Geçersiz dosya formatı!", 400
        except Exception as e:
            print(f"Error occurred: {e}")  # Detaylı hata mesajı loga yazılacak
            return f"Sunucu hatası: {e}", 500
    return render_template('index.html')

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Heroku'nun atadığı portu kullan
    app.run(host="0.0.0.0", port=port, debug=True)  # Debug modunu aktif ettik
