from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow as tf
from tensorflow.python.keras.utils.layer_utils import count_params  
from keras.utils import plot_model
import wandb
from utils import build_model, preprocess, get_dataset

def train(config=None, pretrain=False):

    if sweep:
         wandb.init()
         # Get hyperparameters
         config = wandb.config

    # Define constants
    IMG_WIDTH, IMG_HEIGHT = config['resolution'], config['resolution']

    auc = tf.keras.metrics.AUC(num_thresholds=200, name='PR-AUC', curve='PR')
    final_learning_rate = 1e-5
    learning_rate_decay_factor = (final_learning_rate / config['lr'])**(1/config['epochs'])

    if pretrain:
        MODEL_PATH = './C3/pretrained/cifar10_pretrained.h5'

        es_cback = EarlyStopping(monitor='val_accuracy', mode='max', patience=7, min_delta=0.0001)
        checkpoint_cback = ModelCheckpoint(filepath=MODEL_PATH, mode='max', monitor='val_accuracy', save_best_only=True, save_weights_only=True)
        cbacks = [es_cback, checkpoint_cback]

        train_dataset, test_dataset = get_dataset(config['batch_size'])

        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=config['lr'],
            decay_steps=int(len(train_dataset)/config['batch_size']),
            decay_rate=learning_rate_decay_factor,
            staircase=True)

        model = build_model(config, num_classes=10)  # Ensure model is built for 10 classes

        # Compile the model
        model.compile(optimizer=Adam(learning_rate=lr_schedule, weight_decay=config['l2']),
                    loss='categorical_crossentropy',
                    metrics=['accuracy', auc])
        
        model.fit(train_dataset, 
                  batch_size=config['batch_size'], 
                  epochs=config['epochs'], 
                  validation_data=test_dataset,
                  callbacks=cbacks)

        # Load the trained model
        model.load_weights(MODEL_PATH)

        # Modify the final layer to have 8 channels
        x = model.layers[-4].output
        x = layers.Conv2D(8, kernel_size=3, strides=1, padding='same')(x)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Activation('softmax')(x)
        model = Model(inputs=model.input, outputs=x)

        # Update model path
        MODEL_PATH = './C3/pretrained/model_finetuned.h5'
    else:
        # Update model path
        MODEL_PATH = './C3/pretrained/model_no_pretrain.h5'
        # Load model with 8-channel last layer
        model = build_model(config, num_classes=8)

    es_cback = EarlyStopping(monitor='val_accuracy', mode='max', patience=15, min_delta=0.0001)
    checkpoint_cback = ModelCheckpoint(filepath=MODEL_PATH, mode='max', monitor='val_accuracy', save_best_only=True, save_weights_only=True)
    cbacks = [es_cback, checkpoint_cback]

    DATASET_DIR = '/export/home/mcv/datasets/C3/MIT_small_train_1/'
    # Define the data generator for data augmentation and preprocessing
    train_data_generator = ImageDataGenerator(
        preprocessing_function=lambda x: x/255.0,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        vertical_flip=False,
    )

    # Load and preprocess the training and validation datasets
    train_dataset = train_data_generator.flow_from_directory(
        directory=DATASET_DIR + '/train/',
        target_size=(IMG_WIDTH, IMG_HEIGHT),
        batch_size=config['batch_size'],
        class_mode='categorical',
        shuffle=True,
    )

    val_dataset = train_data_generator.flow_from_directory(
        directory=DATASET_DIR + '/test/',
        target_size=(IMG_WIDTH, IMG_HEIGHT),
        batch_size=config['batch_size'],
        class_mode='categorical',
        shuffle=True,
    )

    # Define the data generator for preprocessing (no augmentation for test data)
    test_data_generator = ImageDataGenerator(preprocessing_function=lambda x: x/255.0)

    # Load and preprocess the test dataset
    test_dataset = test_data_generator.flow_from_directory(
        directory= '/export/home/mcv/datasets/C3/MIT_split/test/',
        target_size=(IMG_WIDTH, IMG_HEIGHT),
        batch_size=config['batch_size'],
        class_mode='categorical',
        shuffle=False  # No need to shuffle the test data
    )

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=config['lr'],
        decay_steps=int(train_dataset.samples/config['batch_size']),
        decay_rate=learning_rate_decay_factor,
        staircase=True)

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=lr_schedule, weight_decay=config['l2']),
                loss='categorical_crossentropy',
                metrics=['accuracy', auc])
                
    # Train the model
    history = model.fit(
        train_dataset,
        epochs=config['epochs'],
        validation_data=val_dataset,
        callbacks=cbacks
    )
    
    for epoch in range(len(history.history['loss'])):
        wandb.log({
            'train_loss': history.history['loss'][epoch],
            'train_accuracy': history.history['accuracy'][epoch],
            'val_loss': history.history['val_loss'][epoch],
            'val_accuracy': history.history['val_accuracy'][epoch]
            })
    wandb.log({'n_params': count_params(model.trainable_weights)})

    # Load the trained model
    model.load_weights(MODEL_PATH)

    # Evaluate the model on the test data
    loss, acc, auc = model.evaluate(test_dataset)

    print(f"Test Loss: {loss}")
    print(f"Test Accuracy: {acc}")
    print(f"Test AUC: {auc}")
    wandb.finish()


sweep = False
if sweep:
    sweep_id = "c3-mcv/cnn2/p6drmh0y"
    wandb.agent(sweep_id, train, count=2)
else:
    config = {
        'resolution': 256,
        'lr': 1e-3,
        'batch_size': 32,
        'epochs': 50,
        'activation': 'gelu',
        'optimizer_type': 'adam',
        'dropout': 0.25,
        'use_batch_norm': True,
        'l2': 0.001,
        'resolution': 256,
        'n_conv_blocks': 2,
        'filters_0': 64,
        'filters_sq_1': 64,
        'filters_ex_1': 128,
        'filters_sq_2': 64,
        'filters_ex_2': 512,
    }

    # Initialize wandb with a sample configuration
    wandb.init(project='cnn2', entity='c3-mcv', config=config)

    # Train the model
    train(config, pretrain=True)