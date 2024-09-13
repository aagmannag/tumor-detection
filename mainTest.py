import cv2
from keras.models import load_model
from PIL import Image
import numpy as np

model = load_model('BrainTumor10Epochsclassifier.h5')

image = cv2.imread('pred\pred5.jpg')
img = Image.fromarray(image)
img = img.resize((64, 64))
img = np.array(img)

in_img = np.expand_dims(img, axis=0)

result = model.predict(in_img)
predicted_class = np.argmax(result, axis=1)

print(predicted_class[0])