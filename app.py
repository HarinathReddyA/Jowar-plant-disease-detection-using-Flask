import base64
from flask import Flask , render_template ,request,Response
import os
import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
import matplotlib.image as mpimg
from tensorflow.keras.layers import LeakyReLU

import sqlite3


app = Flask(__name__)



#def get_model():
#    global model
#    model = load_model('jowar_model3.h5')
 #   print("Model Loaded!")

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)
new_model = keras.models.load_model("jowar_model3/jowar_model3")
#new_model2 = keras.models.load_model("jowar_model3.h5")
UPLOAD_FOLDER = "D:/Capstone/project/static"

label_dict = {0: 'Anthracnose', 1: 'Healthy', 2: 'Leaf Blight'}


def preprocessing_image(path):
    t = 80
    ygt = 35
    img = mpimg.imread(path)
    orange_red = cv2.inRange(img, np.array(
        [255 - t, 69 - t, 0 - t]), np.array([255 + t, 69 + t, 0 + t]))
    dark_orange = cv2.inRange(img, np.array(
        [255 - t, 140 - t, 0 - t]), np.array([255 + t, 140 + t, 0 + t]))
    brown = cv2.inRange(img, np.array(
        [165 - t, 42 - t, 42 - t]), np.array([165 + t, 42 + t, 42 + t]))
    yellow = cv2.inRange(img, np.array(
        [154 - ygt, 205 - ygt, 50 - ygt]), np.array([154 + ygt, 205 + ygt, 50 + ygt]))
    combined_mask = orange_red + dark_orange + yellow + brown
    masked_image = cv2.bitwise_and(img, img, mask=combined_mask)
    compression_size = (250, 250)
    resized_img = cv2.resize(
        masked_image, compression_size, interpolation=cv2.INTER_NEAREST)
    preprocessed_img = resized_img[np.newaxis, :]
    return preprocessed_img


def model_predict(prepro_img):

    predict_variable = new_model.predict(prepro_img,)
    label = np.argmax(predict_variable)

    if label == 1:
        return "It is a " + label_dict[label] + " Leaf"

    else:
        return "Processed Leaf is daignosed with " + label_dict[label] + " disease"


pred = model_predict(preprocessing_image("D:/Capstone/project/static/22.jpg"))


@app.route('/predict', methods=['GET', 'POST'])
def upload_predict():
    if request.method == 'POST':
        image_file = request.files['image']
        if image_file:
            image_location = os.path.join(UPLOAD_FOLDER, image_file.filename)
            image_file.save(image_location)
            return render_template('predict2.html', prediction=model_predict(preprocessing_image(image_location)),image = image_file.filename)
    return render_template('predict2.html', prediction=0, image = None)


@app.route('/',methods=['POST','GET'])
def register1():
    return render_template('Register.html')


@app.route('/register', methods=['POST','GET'])
def Register():
    if request.method == 'POST':
        uname_ = request.form['usrnm']
        email_ = (request.form['email']).lower()
        pass_ = request.form['psw']
        cpass_ = request.form['psw1']
        connection = sqlite3.connect('site.db')
        cursor = connection.cursor()
        cursor.execute('SELECT email FROM user  WHERE email=?', (email_,))
        exsisted_email = cursor.fetchone()
        print(exsisted_email)
        if(exsisted_email):
            print("unsucessfully")
            return render_template('Register.html')

        else:
            print("Registered values successfully...")
            cursor.execute('INSERT INTO user VALUES(?,?,?)', (uname_, email_, pass_))
            connection.commit()    
            cursor.close()
            return render_template('login.html')
    else:
        return render_template('Register.html')


@app.route('/login', methods=['POST', 'GET'])
def login():
    if request.method == 'POST':
        email_ = request.form['email']
        pass_ = request.form['psw']
        connection = sqlite3.connect('site.db')
        cursor = connection.cursor()
        cursor.execute('SELECT email FROM user')
        data_taken = cursor.fetchall()
        if email_ not in np.reshape(data_taken, (1, len(data_taken)))[0]:
            return render_template('Register.html')
        else:
            cursor.execute('SELECT passw FROM user  WHERE email=?', (email_,))
            pass1 = cursor.fetchone()
            if pass_ == pass1[0]:
                return render_template('index.html')
            else:
                return render_template('login.html')
    else:
        return render_template('login.html')
   





#Rendering to about us page
@app.route('/aboutus')
def aboutus():
    return render_template('about.html')


@app.route('/index')
def homepage():
    return render_template('index.html')


@app.route('/predict')
def predict():
    return render_template('predict2.html')

@app.route('/products')
def products():
    return render_template('products.html')


if __name__ == '__main__':
    app.run( debug=True)
