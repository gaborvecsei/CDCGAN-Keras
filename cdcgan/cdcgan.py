import math
import os

import numpy as np
from PIL import Image
from keras import models, layers, optimizers, utils
from keras.datasets import mnist
from keras.layers import Dense, Input
from keras.layers import Reshape
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.convolutional import UpSampling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.normalization import BatchNormalization
from tqdm import tqdm

ACTIVATION = "tanh"


def generator_model():
    input_z = Input((100,))
    x = Dense(1024)(input_z)
    x = layers.Activation(ACTIVATION)(x)
    x = Dense(128 * 7 * 7)(x)
    x = BatchNormalization()(x)
    r_z = Reshape((7, 7, 128), input_shape=(128 * 7 * 7,))(x)

    input_c = layers.Input((100,))
    x = Dense(1024)(input_c)
    x = layers.Activation(ACTIVATION)(x)
    x = Dense(128 * 7 * 7)(x)
    x = BatchNormalization()(x)
    r_c = Reshape((7, 7, 128), input_shape=(128 * 7 * 7,))(x)

    concat_z_c = layers.Concatenate()([r_z, r_c])

    x = UpSampling2D(size=(2, 2))(concat_z_c)
    x = Conv2D(64, (5, 5), padding='same')(x)
    x = layers.Activation(ACTIVATION)(x)
    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(1, (5, 5), padding='same')(x)
    x = layers.Activation(ACTIVATION)(x)
    model = models.Model(inputs=[input_z, input_c], outputs=x)
    return model


def discriminator_model():
    input_gen_image = Input((28, 28, 1))
    conv_1_image = Conv2D(64, (5, 5), padding='same')(input_gen_image)
    act_1_image = layers.Activation(ACTIVATION)(conv_1_image)
    pool_1_image = MaxPooling2D(pool_size=(2, 2))(act_1_image)
    conv_2_image = Conv2D(128, (5, 5))(pool_1_image)
    act_2_image = layers.Activation(ACTIVATION)(conv_2_image)
    pool_2_image = MaxPooling2D(pool_size=(2, 2))(act_2_image)

    input_c = layers.Input((100,))
    dense_1_c = Dense(1024)(input_c)
    act_1_c = layers.Activation(ACTIVATION)(dense_1_c)
    dense_2_c = Dense(5 * 5 * 1)(act_1_c)
    bn_c = BatchNormalization()(dense_2_c)
    reshaped_c = Reshape((5, 5, 1))(bn_c)

    concat = layers.Concatenate()([pool_2_image, reshaped_c])

    x = Flatten()(concat)
    x = Dense(1024)(x)
    x = layers.Activation(ACTIVATION)(x)
    x = Dense(1)(x)
    x = Activation('sigmoid')(x)
    model = models.Model(inputs=[input_gen_image, input_c], outputs=x)
    return model


def generator_containing_discriminator(g, d):
    input_z = layers.Input((100,))
    input_c = layers.Input((100,))
    gen_image = g([input_z, input_c])
    d.trainable = False
    is_real = d([gen_image, input_c])
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


def generate_images(gen, nb_images: int, label: int):
    noise = np.random.uniform(0, 1, size=(nb_images, 100))
    label_batch = np.zeros((nb_images, 100))
    label_batch[:, label] = 1
    generated_images = gen.predict([noise, label_batch], verbose=0)
    return generated_images


def generate_mnist_image_grid(gen):
    generated_images = []

    for i in range(10):
        noise = np.random.uniform(0, 1, size=(10, 100))
        label_input = np.zeros((10, 100))
        label_input[:, i] = 1
        gen_images = gen.predict([noise, label_input], verbose=0)
        generated_images.extend(gen_images)

    generated_images = np.array(generated_images)
    image_grid = combine_images(generated_images)
    return image_grid


def save_generated_image(image, epoch, iteration, folder_path):
    if not os.path.isdir(folder_path):
        os.makedirs(folder_path)

    image = image * 127.5 + 127.5
    file_path = "{0}/{1}_{2}.png".format(folder_path, epoch, iteration)
    Image.fromarray(image.astype(np.uint8)).save(file_path)


def train(batch_size):
    (X_train, y_train), (_, _) = mnist.load_data()
    X_train = (X_train.astype(np.float32) - 127.5) / 127.5
    X_train = X_train[:, :, :, None]

    y_train = utils.to_categorical(y_train, 100)

    print("Generator:")
    G = generator_model()
    G.summary()

    print("Discriminator:")
    D = discriminator_model()
    D.summary()

    print("Combined:")
    GD = generator_containing_discriminator(G, D)
    GD.summary()

    optimizer = optimizers.Adam(0.0002, 0.5)

    G.compile(loss='binary_crossentropy', optimizer=optimizer)
    GD.compile(loss='binary_crossentropy', optimizer=optimizer)
    D.trainable = True
    D.compile(loss='binary_crossentropy', optimizer=optimizer)

    for epoch in range(100):
        nb_of_iterations = int(X_train.shape[0] / batch_size)

        pbar = tqdm(desc="Epoch: {0}".format(epoch), total=X_train.shape[0])
        g_losses = []
        d_losses = []

        for i in range(nb_of_iterations):
            noise = np.random.uniform(0, 1, size=(batch_size, 100))

            image_batch = X_train[i * batch_size:(i + 1) * batch_size]
            label_batch = y_train[i * batch_size:(i + 1) * batch_size]

            generated_images = G.predict([noise, label_batch], verbose=0)

            if i % 20 == 0:
                combined_image = combine_images(generated_images)
                save_generated_image(combined_image, epoch, i, "../images/train_generated_images")

            if i % 20 == 0:
                image_grid = generate_mnist_image_grid(G)
                save_generated_image(image_grid, epoch, i, "../images/generated_mnist_images")

            X = np.concatenate((image_batch, generated_images))
            y = [1] * batch_size + [0] * batch_size
            label_batches_for_discriminator = np.concatenate((label_batch, label_batch))
            d_loss = D.train_on_batch([X, label_batches_for_discriminator], y)
            d_losses.append(d_loss)
            noise = np.random.uniform(0, 1, (batch_size, 100))
            D.trainable = False
            g_loss = GD.train_on_batch([noise, label_batch], [1] * batch_size)
            D.trainable = True
            g_losses.append(g_loss)
            pbar.update(batch_size)

        pbar.close()
        print("\nD loss: {0}, G loss: {1}".format(np.mean(d_losses), np.mean(g_losses)))


if __name__ == "__main__":
    train(batch_size=128)
