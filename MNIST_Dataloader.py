import numpy as np # linear algebra
import struct
from array import array
from os.path import join

class MNIST_Dataloader: 
    def __init__(self): 
        self.training_labels_path = "./Data/train_labels_idx3-ubyte"
        self.training_images_path = "./Data/train-images-idx3-ubyte"
        self.test_labels_path = "./Data/t10k-labels-idx1-ubyte"
        self.test_images_path = "./Data/t10k-images-idx3-ubyte"
        

    def read_image_labels(self, labels_path, images_path): 
        labels = []
        images = []
        with open(labels_path, 'rb') as labels_file:
            magic, size = struct.unpack(">II", labels_file.read(8))
            labels_data = array("B", labels_file.read())

        with open(images_path, 'rb') as images_file:
            magic, size, rows, cols = struct.unpack(">IIII", images_file.read(16))
            image_data = array("B", images_file.read())

        for i in range(size):
            images.append([0] * rows * cols)

        for i in range(size): 
            image = np.array(image_data[i*rows*cols: (i+1)*rows*cols])
            image = image.reshape(28, 28)
            images[i][:] = image

        return images, labels

    def get_test_data(self):
        x_test, y_test = self.read_images_labels(self.test_images_path, self.test_labels_path)
        return x_test, y_test

    def get_train_data(self):
        x_train, y_train = self.read_images_labels(self.training_images_path, self.training_labels_path)
        return x_train, y_train





    