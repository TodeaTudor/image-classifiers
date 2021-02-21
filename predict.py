import numpy as np
import tensorflow as tf
from tensorflow import keras
import sys
img_height = 180
img_width = 180

model_path = sys.argv[1]
img_path = model_path = sys.argv[2]
model = tf.keras.models.load_model(model_path)


img = keras.preprocessing.image.load_img(
    img_path, target_size=(img_height, img_width)
)
img_array = keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])
class_names = ['cat', 'dog']

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)

