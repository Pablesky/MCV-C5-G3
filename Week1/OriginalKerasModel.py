os.environ["KERAS_BACKEND"] = "tensorflow"  # Or "jax" or "torch"!
import os,sys
import numpy as np
from sklearn.feature_extraction import image
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Reshape, Input, Flatten, Conv2D, MaxPooling2D, concatenate, Add, DepthwiseConv2D, BatchNormalization, ReLU, GlobalAveragePooling2D, Activation, LeakyReLU
from keras.activations import elu, selu
import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Reshape, Input, Flatten, Conv2D, MaxPooling2D, concatenate, Add, DepthwiseConv2D, BatchNormalization, ReLU, GlobalAveragePooling2D, Activation, LeakyReLU, Dropout
from keras.models import Model
from keras.activations import elu
from keras.layers.experimental.preprocessing import Rescaling, RandomFlip, Normalization
from keras.utils import plot_model
from tensorflow.keras.regularizers import l1, l2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import optuna 
from tensorflow.keras.preprocessing import image
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
import os,sys
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Reshape, Input, Flatten, Conv2D, MaxPooling2D, concatenate, Add, DepthwiseConv2D, BatchNormalization, ReLU, GlobalAveragePooling2D, Activation, LeakyReLU
from keras.activations import elu, selu

def load_data_augmented(dataset_dir, img_height, img_width, batch_size):
    train_data_generator = ImageDataGenerator(featurewise_center=False,
                                              samplewise_center=False,
                                              featurewise_std_normalization=False,
                                              samplewise_std_normalization=False,
                                              rotation_range=0,
                                              width_shift_range=0.2,
                                              height_shift_range=0.,
                                              shear_range=0.2,
                                              brightness_range=[0.7, 1.3],
                                              zoom_range=0.,
                                              fill_mode='nearest',
                                              horizontal_flip=True,
                                              vertical_flip=False,
                                              rescale= 1./255)

    validation_data_generator = ImageDataGenerator(rescale= 1./255)

    train_dataset = train_data_generator.flow_from_directory(
        directory=dataset_dir+'/train/',
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True)

    validation_dataset = validation_data_generator.flow_from_directory(
        directory=dataset_dir+'/test/',
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True)

    return train_dataset, validation_dataset

def optimizers_config(optimizer, lr):
    if optimizer == 'adam':
        return tf.keras.optimizers.Adam(learning_rate=lr)
    elif optimizer == 'sgd':
        return tf.keras.optimizers.SGD(learning_rate=lr)
    elif optimizer == 'nadam':
        return tf.keras.optimizers.Nadam(learning_rate=lr)
    elif optimizer == 'adadelta':
        return tf.keras.optimizers.Adadelta(learning_rate=lr)

def start1(inputs, activation, regularizer, initializer):
    x = Conv2D(16, (3, 3), strides=(2, 2), padding=1, kernel_regularizer=regularizer, kernel_initializer=initializer)(inputs)
    x = BatchNormalization()(x)
    x = activate_layer(x, activation)
    return x

def separable_block(x, filters, strides, activation, regularizer, initializer):
    x = DepthwiseConv2D((3, 3), strides=strides, padding='same', depthwise_initializer=initializer)(x)
    x = BatchNormalization()(x)
    x = activate_layer(x, activation)

    x = Conv2D(filters, (1, 1), strides=(1, 1), padding='same', kernel_regularizer=regularizer, kernel_initializer=initializer)(x)
    x = BatchNormalization()(x)
    x = activate_layer(x, activation)
    return x

def activate_layer(x, activation):
    if activation == 'relu':
        return ReLU()(x)
    if activation == 'leakyrelu':
        return LeakyReLU()(x)
    if activation == 'elu':
        return elu(x)
    if activation == 'selu':
        return selu(x)

def best_model(input_shape, classes, activation, regularizer, initializer):

    inputs = Input(shape=input_shape)

    x = start1(inputs, activation, regularizer, initializer)

    x = separable_block(x, 32, (1, 1), activation, regularizer, initializer)
    x = separable_block(x, 64, (2, 2), activation, regularizer, initializer)
    x = separable_block(x, 32, (1, 1), activation, regularizer, initializer)
    x = separable_block(x, 64, (2, 2), activation, regularizer, initializer)
    x = separable_block(x, 128, (1, 1), activation, regularizer, initializer)
    x = separable_block(x, 32, (2, 2), activation, regularizer, initializer)
    x = separable_block(x, 128, (1, 1), activation, regularizer, initializer)
   
    x = GlobalAveragePooling2D()(x)

    x = Dense(classes, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=x)
    return model

BATCH_SIZE= 8
ACTIVATION= 'leakyrelu'
REGULARIZER = l2(0.01)
INITIALIZER = 'he_normal'
IMG_HEIGHT=360
IMG_WIDTH = 360
OPTIMIZER='nadam'
LEARNING_RATE=0.001
EPOCHS = 200
DATASET_DIR = '/ghome/mcv/datasets/C3/MIT_small_train_1'
INPUT_SHAPE = (IMG_HEIGHT, IMG_WIDTH, 3)
CLASSES = 8
FILE = 's7_1F3'

train_dataset, validation_dataset = load_data_augmented(DATASET_DIR, IMG_HEIGHT, IMG_WIDTH, BATCH_SIZE)

model = best_model(INPUT_SHAPE, CLASSES, ACTIVATION, REGULARIZER, INITIALIZER)

opt= optimizers_config(OPTIMIZER, LEARNING_RATE)

model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

checkpoint = ModelCheckpoint('best_model_'+FILE+'.keras', save_best_only=True, monitor='val_accuracy', mode='max')

history = model.fit(
    train_dataset,
    epochs=EPOCHS,
    validation_data=validation_dataset,
    callbacks=[checkpoint], 
    verbose = 0
)

# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig(FILE+'_accuracy.jpg')
plt.close()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig(FILE+'_loss.jpg')
plt.close()
print(max(history.history['val_accuracy']))
print(f'ratio {max(history.history["val_accuracy"]) / (model.count_params()/100000)}')
print(FILE)