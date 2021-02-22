from fpidataset import Fpidataset
import numpy as np

dataset = Fpidataset()

(x_train, y_train), (x_test, y_test) = dataset.load_data()
x = np.concatenate((x_train, x_test))
y = np.concatenate((y_train, y_test))

print("shape: ", x.shape)

x = x.reshape((x.shape[0], -1))

print("shape: ", x.shape)

x = np.divide(x, 255.)



