import math

import numpy as np
from PIL import Image
from keras.datasets import mnist
from keras import models, layers, optimizers, activations, utils
from keras.layers import Dense, Input
from keras.layers import Reshape
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.convolutional import UpSampling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from tqdm import tqdm
from keras.optimizers import SGD


def generator_model():
    input_z = Input((100,))
    x = Dense(1024)(input_z)
    x = layers.Activation("tanh")(x)
    x = Dense(128 * 7 * 7)(x)
    x = BatchNormalization()(x)
    r_z = Reshape((7, 7, 128), input_shape=(128 * 7 * 7,))(x)

    input_c = layers.Input((100,))
    x = Dense(1024)(input_c)
    x = layers.Activation("tanh")(x)
    x = Dense(128 * 7 * 7)(x)
    x = BatchNormalization()(x)
    r_c = Reshape((7, 7, 128), input_shape=(128 * 7 * 7,))(x)

    concat_z_c = layers.Concatenate()([r_z, r_c])

    x = UpSampling2D(size=(2, 2))(concat_z_c)
    x = Conv2D(64, (5, 5), padding='same')(x)
    x = layers.Activation("tanh")(x)
    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(1, (5, 5), padding='same')(x)
    x = layers.Activation("tanh")(x)
    model = models.Model(inputs=[input_z, input_c], outputs=x)
    return model


def discriminator_model():
    input_gen_image = Input((28, 28, 1))
    x = Conv2D(64, (5, 5), padding='same')(input_gen_image)
    x = layers.Activation("tanh")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(128, (5, 5))(x)
    x = layers.Activation("tanh")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Flatten()(x)
    x = Dense(1024)(x)
    x = layers.Activation("tanh")(x)
    x = Dense(1)(x)
    x = Activation('sigmoid')(x)
    model = models.Model(inputs=input_gen_image, outputs=x)
    return model


def generator_containing_discriminator(g, d):
    input_z = layers.Input((100,))
    input_c = layers.Input((100,))
    gen_image = g([input_z, input_c])
    d.trainable = False
    is_real = d(gen_image)
    model = models.Model(inputs=[input_z, input_c], outputs=is_real)

    return model


def combine_images(generated_images):
    num = generated_images.shape[0]
    width = int(math.sqrt(num))
    height = int(math.ceil(float(num) / width))
    shape = generated_images.shape[1:3]
    image = np.zeros((height * shape[0], width * shape[1]),
                     dtype=generated_images.dtype)
    for index, img in enumerate(generated_images):
        i = int(index / width)
        j = index % width
        image[i * shape[0]:(i + 1) * shape[0], j * shape[1]:(j + 1) * shape[1]] = \
            img[:, :, 0]
    return image


def train(BATCH_SIZE):
    (X_train, y_train), (_, _) = mnist.load_data()
    X_train = (X_train.astype(np.float32) - 127.5) / 127.5
    X_train = X_train[:, :, :, None]

    y_train = utils.to_categorical(y_train, 100).astype(X_train.dtype)

    print("Discriminator:")
    D = discriminator_model()
    D.summary()

    print("Generator:")
    G = generator_model()
    G.summary()

    print("Combined:")
    GD = generator_containing_discriminator(G, D)
    GD.summary()

    optimizer = optimizers.Adam(0.0002, 0.5)

    G.compile(loss='binary_crossentropy', optimizer=optimizer)
    GD.compile(loss='binary_crossentropy', optimizer=optimizer)
    D.trainable = True
    D.compile(loss='binary_crossentropy', optimizer=optimizer)

    for epoch in range(100):
        nb_of_iterations = int(X_train.shape[0] / BATCH_SIZE)

        pbar = tqdm(desc="Epoch: {0}".format(epoch), total=X_train.shape[0])
        g_losses = []
        d_losses = []

        for i in range(nb_of_iterations):
            noise = np.random.uniform(-1, 1, size=(BATCH_SIZE, 100))

            image_batch = X_train[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
            label_batch = y_train[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]

            generated_images = G.predict([noise, label_batch], verbose=0)
            if i % 20 == 0:
                image = combine_images(generated_images)
                image = image * 127.5 + 127.5
                Image.fromarray(image.astype(np.uint8)).save("../images/" +
                                                             str(epoch) + "_" + str(i) + ".png")
            X = np.concatenate((image_batch, generated_images))
            y = [1] * BATCH_SIZE + [0] * BATCH_SIZE
            d_loss = D.train_on_batch(X, y)
            d_losses.append(d_loss)
            noise = np.random.uniform(-1, 1, (BATCH_SIZE, 100))
            D.trainable = False
            g_loss = GD.train_on_batch([noise, label_batch], [1] * BATCH_SIZE)
            D.trainable = True
            g_losses.append(g_loss)
            pbar.update(BATCH_SIZE)

        pbar.close()
        print("\nD loss: {0}, G loss: {1}".format(np.mean(d_losses), np.mean(g_losses)))


if __name__ == "__main__":
    train(BATCH_SIZE=128)
