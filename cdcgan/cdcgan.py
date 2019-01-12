import math
import os

import cv2
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from keras import backend as K
from keras import models, layers, optimizers, utils
from keras import datasets
from swiss_army_tensorboard import tfboard_loggers
from tqdm import tqdm

ACTIVATION = layers.Activation("tanh")


def generator_model():
    input_z = layers.Input((100,))
    dense_z_1 = layers.Dense(1024)(input_z)
    act_z_1 = ACTIVATION(dense_z_1)
    dense_z_2 = layers.Dense(128 * 7 * 7)(act_z_1)
    bn_z_1 = layers.BatchNormalization()(dense_z_2)
    reshape_z = layers.Reshape((7, 7, 128), input_shape=(128 * 7 * 7,))(bn_z_1)

    input_c = layers.Input((100,))
    dense_c_1 = layers.Dense(1024)(input_c)
    act_c_1 = ACTIVATION(dense_c_1)
    dense_c_2 = layers.Dense(128 * 7 * 7)(act_c_1)
    bn_c_1 = layers.BatchNormalization()(dense_c_2)
    reshape_c = layers.Reshape((7, 7, 128), input_shape=(128 * 7 * 7,))(bn_c_1)

    concat_z_c = layers.Concatenate()([reshape_z, reshape_c])

    up_1 = layers.UpSampling2D(size=(2, 2))(concat_z_c)
    conv_1 = layers.Conv2D(64, (5, 5), padding='same')(up_1)
    act_1 = ACTIVATION(conv_1)
    up_2 = layers.UpSampling2D(size=(2, 2))(act_1)
    conv_2 = layers.Conv2D(1, (5, 5), padding='same')(up_2)
    act_2 = layers.Activation("tanh")(conv_2)
    model = models.Model(inputs=[input_z, input_c], outputs=act_2)
    return model


def discriminator_model():
    input_gen_image = layers.Input((28, 28, 1))
    conv_1_image = layers.Conv2D(64, (5, 5), padding='same')(input_gen_image)
    act_1_image = ACTIVATION(conv_1_image)
    pool_1_image = layers.MaxPooling2D(pool_size=(2, 2))(act_1_image)
    conv_2_image = layers.Conv2D(128, (5, 5))(pool_1_image)
    act_2_image = ACTIVATION(conv_2_image)
    pool_2_image = layers.MaxPooling2D(pool_size=(2, 2))(act_2_image)

    input_c = layers.Input((100,))
    dense_1_c = layers.Dense(1024)(input_c)
    act_1_c = ACTIVATION(dense_1_c)
    dense_2_c = layers.Dense(5 * 5 * 1)(act_1_c)
    bn_c = layers.BatchNormalization()(dense_2_c)
    reshaped_c = layers.Reshape((5, 5, 1))(bn_c)

    concat = layers.Concatenate()([pool_2_image, reshaped_c])

    flat = layers.Flatten()(concat)
    dense_1 = layers.Dense(1024)(flat)
    act_1 = ACTIVATION(dense_1)
    dense_2 = layers.Dense(1)(act_1)
    act_2 = layers.Activation('sigmoid')(dense_2)
    model = models.Model(inputs=[input_gen_image, input_c], outputs=act_2)
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
    num_images = generated_images.shape[0]
    new_width = int(math.sqrt(num_images))
    new_height = int(math.ceil(float(num_images) / new_width))
    grid_shape = generated_images.shape[1:3]
    grid_image = np.zeros((new_height * grid_shape[0], new_width * grid_shape[1]), dtype=generated_images.dtype)
    for index, img in enumerate(generated_images):
        i = int(index / new_width)
        j = index % new_width
        grid_image[i * grid_shape[0]:(i + 1) * grid_shape[0], j * grid_shape[1]:(j + 1) * grid_shape[1]] = \
            img[:, :, 0]
    return grid_image


def generate_noise(shape: tuple):
    noise = np.random.uniform(0, 1, size=shape)
    return noise


def generate_images(gen, nb_images: int, label: int):
    noise = generate_noise((nb_images, 100))
    label_batch = np.zeros((nb_images, 100))
    label_batch[:, label] = 1
    generated_images = gen.predict([noise, label_batch], verbose=0)
    return generated_images


def generate_mnist_image_grid(gen, title: str = "Generated images"):
    generated_images = []

    for i in range(10):
        noise = generate_noise((10, 100))
        label_input = np.zeros((10, 100))
        label_input[:, i] = 1
        gen_images = gen.predict([noise, label_input], verbose=0)
        generated_images.extend(gen_images)

    generated_images = np.array(generated_images)
    image_grid = combine_images(generated_images)
    image_grid = inverse_transform_images(image_grid)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.axis("off")
    ax.imshow(image_grid, cmap="gray")
    ax.set_title(title)
    fig.canvas.draw()

    image = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    return image


def save_generated_image(image, epoch, iteration, folder_path):
    if not os.path.isdir(folder_path):
        os.makedirs(folder_path)

    file_path = "{0}/{1}_{2}.png".format(folder_path, epoch, iteration)
    cv2.imwrite(file_path, image.astype(np.uint8))


def transform_images(images: np.ndarray):
    images = (images.astype(np.float32) - 127.5) / 127.5
    return images


def inverse_transform_images(images: np.ndarray):
    images = images * 127.5 + 127.5
    images = images.astype(np.uint8)
    return images


def train(batch_size):
    (X_train, y_train), (_, _) = datasets.mnist.load_data()
    X_train = transform_images(X_train)
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

    # Setup Tensorboard loggers
    tfboard_loggers.TFBoardModelGraphLogger.log_graph("../models/logs", K.get_session())
    loss_logger = tfboard_loggers.TFBoardScalarLogger("../models/logs/loss")
    image_logger = tfboard_loggers.TFBoardImageLogger("../models/logs/generated_images")

    iteration = 0

    nb_of_iterations_per_epoch = int(X_train.shape[0] / batch_size)
    print("Number of iterations per epoch: {0}".format(nb_of_iterations_per_epoch))

    for epoch in range(100):
        pbar = tqdm(desc="Epoch: {0}".format(epoch), total=X_train.shape[0])
        g_losses = []
        d_losses = []

        for i in range(nb_of_iterations_per_epoch):
            noise = generate_noise((batch_size, 100))

            image_batch = X_train[i * batch_size:(i + 1) * batch_size]
            label_batch = y_train[i * batch_size:(i + 1) * batch_size]

            generated_images = G.predict([noise, label_batch], verbose=0)

            if i % 20 == 0:
                image_grid = generate_mnist_image_grid(G, title="Epoch {0}, iteration {1}".format(epoch, iteration))
                save_generated_image(image_grid, epoch, i, "../images/generated_mnist_images_per_iteration")
                image_logger.log_images("generated_mnist_images_per_iteration", [image_grid], iteration)

            X = np.concatenate((image_batch, generated_images))
            y = [1] * batch_size + [0] * batch_size
            label_batches_for_discriminator = np.concatenate((label_batch, label_batch))
            D_loss = D.train_on_batch([X, label_batches_for_discriminator], y)
            d_losses.append(D_loss)
            loss_logger.log_scalar("discriminator_loss", D_loss, iteration)
            noise = generate_noise((batch_size, 100))
            D.trainable = False
            G_loss = GD.train_on_batch([noise, label_batch], [1] * batch_size)
            D.trainable = True
            g_losses.append(G_loss)
            loss_logger.log_scalar("generator_loss", G_loss, iteration)

            pbar.update(batch_size)

            iteration += 1

        # Save a generated image for every epoch
        image_grid = generate_mnist_image_grid(G, title="Epoch {0}".format(epoch))
        save_generated_image(image_grid, epoch, 0, "../images/generated_mnist_images_per_epoch")
        image_logger.log_images("generated_mnist_images_per_epoch", [image_grid], epoch)

        pbar.close()
        print("D loss: {0}, G loss: {1}".format(np.mean(d_losses), np.mean(g_losses)))

        G.save_weights("../models/weights/generator.h5")
        D.save_weights("../models/weights/discriminator.h5")


if __name__ == "__main__":
    train(batch_size=128)
