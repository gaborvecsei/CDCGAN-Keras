import matplotlib

matplotlib.use('Agg')
import numpy as np
from keras import backend as K
from keras import utils as keras_utils
from keras import optimizers
from keras import datasets
from swiss_army_tensorboard import tfboard_loggers
from tqdm import tqdm
from cdcgan import cdcgan_models, cdcgan_utils

BATCH_SIZE = 128
EPOCHS = 100

# Load & Prepare MNIST

(X_train, y_train), (_, _) = datasets.mnist.load_data()
X_train = cdcgan_utils.transform_images(X_train)
X_train = X_train[:, :, :, None]

y_train = keras_utils.to_categorical(y_train, 100)

# Create the models

print("Generator:")
G = cdcgan_models.generator_model()
G.summary()

print("Discriminator:")
D = cdcgan_models.discriminator_model()
D.summary()

print("Combined:")
GD = cdcgan_models.generator_containing_discriminator(G, D)
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

# Model Training

iteration = 0

nb_of_iterations_per_epoch = int(X_train.shape[0] / BATCH_SIZE)
print("Number of iterations per epoch: {0}".format(nb_of_iterations_per_epoch))

for epoch in range(EPOCHS):
    pbar = tqdm(desc="Epoch: {0}".format(epoch), total=X_train.shape[0])

    g_losses_for_epoch = []
    d_losses_for_epoch = []

    for i in range(nb_of_iterations_per_epoch):
        noise = cdcgan_utils.generate_noise(BATCH_SIZE)

        image_batch = X_train[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
        label_batch = y_train[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]

        generated_images = G.predict([noise, label_batch], verbose=0)

        if i % 20 == 0:
            image_grid = cdcgan_utils.generate_image_grid(G,
                                                          title="Epoch {0}, iteration {1}".format(epoch,
                                                                                                  iteration))
            cdcgan_utils.save_generated_image(image_grid, epoch, i, "../images/generated_mnist_images_per_iteration")
            image_logger.log_images("generated_mnist_images_per_iteration", [image_grid], iteration)

        X = np.concatenate((image_batch, generated_images))
        y = [1] * BATCH_SIZE + [0] * BATCH_SIZE
        label_batches_for_discriminator = np.concatenate((label_batch, label_batch))

        D_loss = D.train_on_batch([X, label_batches_for_discriminator], y)
        d_losses_for_epoch.append(D_loss)
        loss_logger.log_scalar("discriminator_loss", D_loss, iteration)

        noise = cdcgan_utils.generate_noise(BATCH_SIZE)
        D.trainable = False
        G_loss = GD.train_on_batch([noise, label_batch], [1] * BATCH_SIZE)
        D.trainable = True
        g_losses_for_epoch.append(G_loss)
        loss_logger.log_scalar("generator_loss", G_loss, iteration)

        pbar.update(BATCH_SIZE)

        iteration += 1

    # Save a generated image for every epoch
    image_grid = cdcgan_utils.generate_image_grid(G, title="Epoch {0}".format(epoch))
    cdcgan_utils.save_generated_image(image_grid, epoch, 0, "../images/generated_mnist_images_per_epoch")
    image_logger.log_images("generated_mnist_images_per_epoch", [image_grid], epoch)

    pbar.close()
    print("D loss: {0}, G loss: {1}".format(np.mean(d_losses_for_epoch), np.mean(g_losses_for_epoch)))

    G.save_weights("../models/weights/generator.h5")
    D.save_weights("../models/weights/discriminator.h5")
