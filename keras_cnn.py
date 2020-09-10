import tensorflow as tf 
from tensorflow import keras 
import numpy as np
import pydot 

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

def get_dataset (training = True):
    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    if training:
        tmp = np.array(train_images)
        return np.expand_dims(tmp, axis=3), np.array(train_labels)
    else:
        tmp = np.array(test_images)
        return np.expand_dims(tmp, axis=3), np.array(test_labels)

def build_model():
    model = keras.Sequential()
    images, labels = get_dataset()
    l = len(images[0])
    b = len(images[0][0])
    w = len(images[0][0][0])
    model.add(keras.layers.Conv2D(filters = 64, kernel_size = 3, activation = 'relu', input_shape=(l, b, w)))
    model.add(keras.layers.Conv2D(filters = 32, kernel_size = 3, activation = 'relu', input_shape=(l, b, w)))
    # Flatten layer
    model.add(keras.layers.Flatten())
    # Dense Layer 1
    model.add(keras.layers.Dense(10, activation = 'softmax'))
    # Compile 
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])    
    keras.utils.plot_model(model, to_file='model.png')
    return model


def train_model(model, train_img, train_lab, test_img, test_lab, T): 
    
    train_labels = keras.utils.to_categorical(train_lab)
    test_labels = keras.utils.to_categorical(test_lab)
    model.fit(x=train_img, y = train_labels, validation_data = (test_img, test_labels), epochs = T)

# predict_label(model, images, index) — takes the trained model and test images, 
# and prints the top 3 most likely labels for the image at the given index, along with their probabilities

def predict_label(model, images, index):

    img = images[index]
    list = model.predict(img.reshape(1, len(img), len(img[0]), len(img[0][0])), verbose = 0)[0].tolist()
    dict = {}
    for i in range(len(class_names)):
        dict[list[i]] = class_names[i]
    list = sorted(dict.keys(), reverse = True)

    for key in list[:3]:
        print(f"{dict[key]}: {round(key*100, 2)}%")

# __________________________________execution_______________________________________

if __name__ == "__main__":
    train_images, train_labels = get_dataset()
    model = build_model()
    val_img, val_lab = get_dataset(False)
    train_model(model, train_images, train_labels, val_img, val_lab, 5)
    
    # print("\n\n\n\n")
    # evaluate_model(model, test_images, test_labels, show_loss=True)
    predict_label(model, val_img, 0)
    predict_label(model, val_img, 9)