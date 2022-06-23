## MODEL ##

from tensorflow.keras import layers, Model
from modules.config import RESIZE

def baseline_model():
    
    # image input parameters
    inputs = layers.Input(shape=(RESIZE[1], RESIZE[0], 1))
    
    # convolusion, activation filer, pool
    x = layers.Conv2D(64, kernel_size=(5, 5))(inputs)
    x = layers.ReLU()(x)
    x = layers.MaxPool2D((2, 2))(x)
    
    # denses layers to coordinates prediction
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(units=256)(x)
    x = layers.ReLU()(x)
    x = layers.Dense(units=2)(x)
    
    model = Model(inputs, outputs=x)
    
    return model

def final_model():
    
    # image input parameters
    inputs = layers.Input(shape=(RESIZE[1], RESIZE[0], 1))
    
    # convolusion, activation filer, pool
    x = layers.Conv2D(32, kernel_size=(5, 5))(inputs)
    x = layers.ReLU()(x)
    x = layers.MaxPool2D((2, 2))(x)
    
    x = layers.Conv2D(64, kernel_size=(3, 3))(x)
    x = layers.ReLU()(x)
    x = layers.MaxPool2D((2, 2))(x)
    
    x = layers.Conv2D(128, kernel_size=(3, 3))(x)
    x = layers.ReLU()(x)
    x = layers.MaxPool2D((2, 2))(x)
    
    x = layers.Conv2D(256, kernel_size=(3, 3))(x)
    x = layers.ReLU()(x)
    x = layers.MaxPool2D((2, 2))(x)
    
    x = layers.Conv2D(512, kernel_size=(3, 3))(x)
    x = layers.ReLU()(x)
    x = layers.MaxPool2D((2, 2))(x)
    
     # denses layers to coordinates prediction
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(units=256)(x)
    x = layers.ReLU()(x)
    x = layers.Dense(units=2)(x)
    
    model = Model(inputs, outputs=x)
    
    return model