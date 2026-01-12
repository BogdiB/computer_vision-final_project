import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os


BASE_DIR = os.path.abspath(os.path.curdir)
IMG_HEIGHT = 128
IMG_WIDTH = 128

model = load_model("rock_paper_scissors_cnn.h5")
class_labels = ['paper', 'rock', 'scissors']


def predict_image(img_path):
    img = image.load_img(BASE_DIR + "\\" + img_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions)

    return class_labels[predicted_class]


# "rps-cv-images/" + "paper"/"rock"/"scissors" + whatever file from there
result = predict_image("rps-cv-images\\rock\\0bioBZYFCXqJIulm.png")
print("Prediction:", result)
