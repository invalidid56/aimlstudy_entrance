import os
import shutil

from keras.layers import Dense, Dropout, GlobalAvgPool2D, Input, RandomFlip, RandomRotation, Rescaling
from keras.callbacks import EarlyStopping, TensorBoard
from keras.optimizers import RMSprop
from keras.losses import CategoricalCrossentropy
from keras.models import Sequential, Model, save_model
from keras.utils import image_dataset_from_directory
from keras.applications import MobileNetV2
import tensorflow as tf

IMG_SIZE = (160, 160)
BATCH_SIZE = 32
NUM_OF_CLASSES = 10
SEED = 1313
IMG_SHAPE = (160, 160, 3)
FINE_TUNE_AT = 100


def main(raw_dir='raw-img', log_dir='log', model_dir='model'):
    #
    # Read Raw Data, Classify
    #
    if os.path.exists(model_dir):
        shutil.rmtree(model_dir)
    os.mkdir(model_dir)

    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.mkdir(log_dir)
    train_set = image_dataset_from_directory(raw_dir,
                                             shuffle=True,
                                             batch_size=BATCH_SIZE,
                                             image_size=IMG_SIZE,
                                             validation_split=0.8,
                                             subset='training',
                                             seed=SEED,
                                             label_mode='categorical'
                                             )
    val_set = image_dataset_from_directory(raw_dir,
                                           shuffle=True,
                                           batch_size=BATCH_SIZE,
                                           image_size=IMG_SIZE,
                                           validation_split=0.2,
                                           subset='validation',
                                           seed=SEED,
                                           label_mode='categorical'
                                           )

    val_batches = tf.data.experimental.cardinality(val_set)
    test_set = val_set.take(val_batches // 5)
    val_set = val_set.skip(val_batches // 5)

    print(train_set, val_set)
    #
    # Preprocess(160*160) Augmentation
    #
    augmentation_preprocessing = Sequential(
        [RandomFlip('horizontal'), RandomRotation(0.2), Rescaling(1. / 127.5, offset=-1)])

    #
    # Train
    #
    base_model = MobileNetV2(input_shape=IMG_SHAPE,
                             include_top=False,
                             weights='imagenet')
    base_model.trainable = True

    for layer in base_model.layers[:FINE_TUNE_AT]:
        layer.trainable = False

    global_average_layer = GlobalAvgPool2D()

    prediction_layer = Dense(NUM_OF_CLASSES, activation='softmax')

    inputs_ = Input(shape=IMG_SHAPE)
    x = augmentation_preprocessing(inputs_)
    x = base_model(x, training=False)
    x = global_average_layer(x)
    x = Dropout(0.2)(x)
    outputs_ = prediction_layer(x)

    model = Model(inputs_, outputs_)
    model.compile(optimizer=RMSprop(lr=0.00001),
                  loss=CategoricalCrossentropy(),
                  metrics=['accuracy'],
                  )

    TB = TensorBoard(log_dir=log_dir)
    ES = EarlyStopping(monitor='val_loss', patience=10)

    _ = model.fit(train_set,
                  epochs=20,
                  validation_data=val_set,
                  callbacks=[TB, ES])

    #
    # Save Model and Estimation
    #

    loss, accuracy = model.evaluate(test_set)
    print("Loss: {0}\nAccuracy: {1}".format(loss, accuracy))

    save_model(model, model_dir)


if __name__ == '__main__':
    main('raw-img')
