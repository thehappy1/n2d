from fpidataset import Fpidataset
from keras.datasets import fashion_mnist
from PIL import Image
import numpy as np

dataset = Fpidataset()

x_train, x_test, y_train, y_test = dataset.load_data()

print(x_train.columns)

img_path = x_train.image_path[0]
print("image path: ",img_path)

img_1 = Image.open(img_path).convert('RGB')
np_array_1 = np.asarray(img_1)

img_2 = Image.open(x_train.image_path[32]).convert('RGB')
np_array_2 = np.asarray(img_2)

x = np_array_1

print("länge: ",len(x))

x = np.append(x,np_array_2)

print(x.shape)
print("länge: ",len(x))

