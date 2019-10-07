from keras.applications.resnet50 import ResNet50, preprocess_input
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint, EarlyStopping
import math
from generate_plot_pics import generate_plot_pics

HEIGHT = 150
WIDTH = 150

class_list = ['bR', 'bN', 'bB', 'bQ', 'bK', 'bP', 'wP', 'wR', 'wN', 'wB', 'wQ', 'wK', '__']

from keras.preprocessing.image import ImageDataGenerator, load_img

TRAIN_DIR = "images_chess_pieces"
BATCH_SIZE = 8
# BATCH_SIZE = 32

train_datagen =  ImageDataGenerator(
      preprocessing_function=preprocess_input,
      # rotation_range=5,
      horizontal_flip=True,
      brightness_range=[0.6, 1.4],
      # zoom_range=0.05,
      # shear_range=0.03,
      # zca_whitening=True,
      validation_split=0.2,
    )

if False:
  generate_plot_pics(train_datagen,load_img("images_chess_pieces/wN/_board_1454.jpg_750_300.jpg"))
  exit()


train_generator = train_datagen.flow_from_directory(TRAIN_DIR, target_size=(HEIGHT, WIDTH), batch_size=BATCH_SIZE, subset='training', class_mode='categorical')
validation_generator = train_datagen.flow_from_directory(TRAIN_DIR, target_size=(HEIGHT, WIDTH), batch_size=BATCH_SIZE, subset='validation', class_mode='categorical')

from keras.layers import Dense, Activation, Flatten, Dropout
from keras.models import Sequential, Model

def build_finetune_model(base_model, dropout, num_classes):
    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    x = Flatten()(x)
    for fc in [1024, 1024]:
        x = Dense(fc, activation='relu')(x)
        x = Dropout(dropout)(x)

    # New softmax layer
    predictions = Dense(num_classes, activation='softmax')(x)

    finetune_model = Model(inputs=base_model.input, outputs=predictions)

    return finetune_model

# dropout = 0.5
dropout = 0.5

base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(HEIGHT, WIDTH, 3))
finetune_model = build_finetune_model(base_model,
                                      dropout=dropout,
                                      num_classes=len(class_list))

from keras.optimizers import SGD, Adam

NUM_EPOCHS = 99999
# num_train_images = 10000
num_train_images = len(train_generator)

# optimizer = Adam(lr=0.00001)
optimizer = Adam(lr=0.000015)
sgd = SGD(lr = 0.01, decay = 1e-6, momentum = 0.9, nesterov = True)

finetune_model.compile(optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

print(finetune_model.summary())

filepath="./checkpoints/" + "ResNet50" + "best_model_weights_val_loss_batch_8_sgd.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', mode='min', save_best_only=True)
early_stopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=8)
callbacks_list = [checkpoint, early_stopping]

history = finetune_model.fit_generator(train_generator,
                                       epochs=NUM_EPOCHS,
                                       workers=8,
                                       steps_per_epoch = math.ceil(train_generator.samples // BATCH_SIZE),
                                       validation_data = validation_generator,
                                       validation_steps = math.ceil(validation_generator.samples // BATCH_SIZE),
                                       shuffle=True,
                                       callbacks=callbacks_list)

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

plot_training(history)

