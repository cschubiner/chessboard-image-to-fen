from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.callbacks import ModelCheckpoint, EarlyStopping
import math

HEIGHT = 150
WIDTH = 150

base_model = ResNet50(weights='imagenet',
                      include_top=False,
                      input_shape=(HEIGHT, WIDTH, 3))

class_list = ['bR', 'bN', 'bB', 'bQ', 'bK', 'bP', 'wP', 'wR', 'wN', 'wB', 'wQ', 'wK', '__']

from keras.preprocessing.image import ImageDataGenerator

TRAIN_DIR = "images_chess_pieces"
BATCH_SIZE = 8

train_datagen =  ImageDataGenerator(
      preprocessing_function=preprocess_input,
      rotation_range=15,
      horizontal_flip=True,
      brightness_range=[0.5, 1.5],
      zoom_range=0.05,
      shear_range=0.05,
      zca_whitening=False,
      validation_split=0.15,
    )

train_generator = train_datagen.flow_from_directory(TRAIN_DIR, target_size=(HEIGHT, WIDTH), batch_size=BATCH_SIZE, subset='training')
validation_generator = train_datagen.flow_from_directory(TRAIN_DIR, target_size=(HEIGHT, WIDTH), batch_size=BATCH_SIZE, subset='validation')

from keras.layers import Dense, Activation, Flatten, Dropout
from keras.models import Sequential, Model

def build_finetune_model(base_model, dropout, fc_layers, num_classes):
    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    x = Flatten()(x)
    for fc in fc_layers:
        # New FC layer, random init
        x = Dense(fc, activation='relu')(x)
        x = Dropout(dropout)(x)

    # New softmax layer
    predictions = Dense(num_classes, activation='softmax')(x)

    finetune_model = Model(inputs=base_model.input, outputs=predictions)

    return finetune_model

FC_LAYERS = [1024, 1024]
dropout = 0.5

finetune_model = build_finetune_model(base_model,
                                      dropout=dropout,
                                      fc_layers=FC_LAYERS,
                                      num_classes=len(class_list))

from keras.optimizers import SGD, Adam

NUM_EPOCHS = 99999
# num_train_images = 10000
num_train_images = len(train_generator)

adam = Adam(lr=0.00001)
finetune_model.compile(adam, loss='categorical_crossentropy', metrics=['accuracy'])

filepath="./checkpoints/" + "ResNet50" + "best_model_weights.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', mode='min', save_best_only=True)
early_stopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=3)
callbacks_list = [checkpoint, early_stopping]

history = finetune_model.fit_generator(train_generator,
                                       epochs=NUM_EPOCHS,
                                       workers=8,
                                       steps_per_epoch = train_generator.samples // BATCH_SIZE,
                                       validation_data = validation_generator,
                                       validation_steps = validation_generator.samples // BATCH_SIZE,
                                       shuffle=True,
                                       callbacks=callbacks_list)

plot_training(history)

# Plot the training and validation loss + accuracy
def plot_training(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(acc))

    plt.plot(epochs, acc, 'r.')
    plt.plot(epochs, val_acc, 'r')
    plt.title('Training and validation accuracy')

    # plt.figure()
    # plt.plot(epochs, loss, 'r.')
    # plt.plot(epochs, val_loss, 'r-')
    # plt.title('Training and validation loss')
    plt.show()

    plt.savefig('acc_vs_epochs.png')
