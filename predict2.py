import tensorflow as tf
from tensorflow import keras
import sys


img_height = 160
img_width = 160

model_path = sys.argv[1]
img_path = model_path = sys.argv[2]
model = tf.keras.models.load_model(model_path)

img = keras.preprocessing.image.load_img(
    img_path, target_size=(img_height, img_width)
)

img_array = keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)

predictions = model.predict(img_array)
predictions = tf.nn.sigmoid(predictions)
class_names = ['church', 'mosque']
chance = predictions.numpy()[0].tolist()
if chance[0] >= 0.5:
    object_class = 'mosque'
    confidence = chance[0]
else:
    object_class = 'church'
    confidence = 1 - chance[0]

print(
    "The image likely belongs to class {} with a probability of {:.6f}%".format(object_class, confidence*100)
)