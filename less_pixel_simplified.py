import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
from numpy.random import randn
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adadelta, Adam
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Reshape, LeakyReLU, Flatten
from qibo import gates, models, set_backend
import argparse


set_backend('tensorflow')

# define the standalone discriminator model
def define_discriminator(n_inputs=16, alpha=0.2, dropout=0.2, lr=0.1):
    model = Sequential()     
    model.add(Dense(200, use_bias=False, input_dim=n_inputs))
    model.add(Reshape((10,10,2)))  
    model.add(Conv2D(64, kernel_size=3, strides=1, padding='same', kernel_initializer='glorot_normal'))
    model.add(LeakyReLU(alpha=alpha))
    model.add(Conv2D(32, kernel_size=3, strides=1, padding='same', kernel_initializer='glorot_normal'))
    model.add(LeakyReLU(alpha=alpha))
    model.add(Conv2D(16, kernel_size=3, strides=1, padding='same', kernel_initializer='glorot_normal'))
    model.add(LeakyReLU(alpha=alpha))
    model.add(Conv2D(8, kernel_size=3, strides=1, padding='same', kernel_initializer='glorot_normal'))
    model.add(Flatten())
    model.add(LeakyReLU(alpha=alpha))
    model.add(Dropout(dropout)) 
    
    model.add(Dense(1, activation='sigmoid'))

    # compile model
    opt = Adadelta(learning_rate=lr)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model
 
# define the combined generator and discriminator model, for updating the generator
def define_cost_gan(params, discriminator, latent_dim, samples, circuit, nqubits, layers, pixels):
    # generate fake samples
    x_fake, y_fake = generate_fake_samples(params, latent_dim, samples, circuit, nqubits, layers, pixels)
    # create inverted labels for the fake samples
    y_fake = np.ones((samples, 1))
    # evaluate discriminator on fake examples
    disc_output = discriminator(x_fake)
    loss = tf.keras.losses.binary_crossentropy(y_fake, disc_output)
    loss = tf.reduce_mean(loss)
    return loss

def set_params(circuit, params, x_input, i, nqubits, layers, latent_dim):
    p = []
    index = 0
    noise = 0
    for l in range(layers):
        for q in range(nqubits):
            p.append(params[index] * x_input[noise][i] + params[index+1])
            index += 2
            noise= (noise+1) % latent_dim
            p.append(params[index] * x_input[noise][i] + params[index+1])
            index += 2
            noise = (noise+1) % latent_dim
    for q in range(nqubits):
        p.append(params[index] * x_input[noise][i] + params[index+1])
        index += 2
        noise= (noise+1) % latent_dim
    circuit.set_parameters(p)

def generate_training_real_samples(dataset, samples):
    if dataset == 'dummy_dataset.npy':
        pixels = 16
        images = np.load('dummy_dataset.npy')
        images = images[:samples]

    elif dataset == '4x4MNIST':
        from scipy.io import loadmat
        pixels = 16
        mat = loadmat('4x4MNIST_Train/4x4MNIST_Train/MNIST_Train_Nox4x4.mat')
        images = mat['V'][:samples]
    elif dataset == '8x8Digits':
        from sklearn.datasets import load_digits
        pixels = 64
        images = load_digits(n_class=1).images
        images = images[:samples]
        images = images.astype(np.float32) / 16.
    else:
        return NotImplementedError("Unknown dataset")

    # normalize each image:
    images = np.array([i/sum(i.flatten()) for i in images])
    # reshape
    images = np.reshape(images, (images.shape[0], pixels))
    return images, pixels

# generate real samples with class labels
def generate_real_samples(X_train, batch_size):
    # generate samples from the distribution
    idx = np.random.randint(X_train.shape[0], size=batch_size)
    imgs = X_train[idx]
    y = np.ones((batch_size, 1))

    return imgs, y

# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, samples):
    # generate points in the latent space
    x_input = randn(latent_dim * samples)
    # reshape into a batch of inputs for the network
    x_input = x_input.reshape(samples, latent_dim)
    return x_input
 
# use the generator to generate fake examples, with class labels
def generate_fake_samples(params, latent_dim, samples, circuit, nqubits, layers, pixels):
    # generate points in latent space
    x_input = generate_latent_points(latent_dim, samples)
    x_input = np.transpose(x_input)
    X = []
    for i in range(pixels):
        X.append([])

    # quantum generator circuit
    for i in range(samples):
        set_params(circuit, params, x_input, i, nqubits, layers, latent_dim)
        circuit()
        for ii in range(pixels):
                X[ii].append(abs(circuit.final_state[ii]**2))

    # shape array
    X = tf.stack([X[i] for i in range(len(X))], axis=1)
    # create class labels
    y = np.zeros((samples, 1))
    return X, y

# train the generator and discriminator
def train(d_model, latent_dim, layers, nqubits, training_samples, circuit, n_epochs, samples, lr, lr_d, dataset, folder):
    d_loss = []
    g_loss = []
    # determine half the size of one batch, for updating the discriminator
    half_samples = int(samples / 2)
    initial_params = tf.Variable(np.random.uniform(-0.15, 0.15, 4*layers*nqubits + 2*nqubits))
    optimizer = tf.optimizers.Adadelta(learning_rate=lr)
    # prepare real samples
    s, pixels = generate_training_real_samples(dataset, training_samples)
    # manually enumerate epochs
    for i in range(n_epochs):
        # prepare real samples
        x_real, y_real = generate_real_samples(s, half_samples)
        #print("x_real", x_real)
        # prepare fake examples
        x_fake, y_fake = generate_fake_samples(initial_params, latent_dim, half_samples, circuit, nqubits, layers,pixels)
        #print("x_fake", x_fake)
        # update discriminator
        d_loss_real, _ = d_model.train_on_batch(x_real, y_real)
        d_loss_fake, _ = d_model.train_on_batch(x_fake, y_fake)
        d_loss.append((d_loss_real + d_loss_fake)/2)
        # update generator
        with tf.GradientTape() as tape:
            loss = define_cost_gan(initial_params, d_model, latent_dim, samples, circuit, nqubits, layers, pixels)
        grads = tape.gradient(loss, initial_params)
        optimizer.apply_gradients([(grads, initial_params)])
        g_loss.append(loss)
        np.savetxt(f"{folder}/PARAMS_{nqubits}_{latent_dim}_{layers}_{training_samples}_{samples}_{lr}_{lr_d}", [initial_params.numpy()], newline='')
        np.savetxt(f"{folder}/dloss_{nqubits}_{latent_dim}_{layers}_{training_samples}_{samples}_{lr}_{lr_d}", [d_loss], newline='')
        np.savetxt(f"{folder}/gloss_{nqubits}_{latent_dim}_{layers}_{training_samples}_{samples}_{lr}_{lr_d}", [g_loss], newline='')
        if i % 500 == 0:
            with open(f"{folder}/ALL_PARAMS_{nqubits}_{latent_dim}_{layers}_{training_samples}_{samples}_{lr}_{lr_d}", "ab") as f:
                np.savetxt(f, [initial_params.numpy()])
        # serialize weights to HDF5
        #discriminator.save_weights(f"less_image_test/discriminator_4pxls_{nqubits}_{latent_dim}_{layers}_{training_samples}_{samples}_{lr}.h5")
    return loss

def build_and_train_model(lr_d=1e-2, lr=1e-2, n_epochs=10, batch_samples=10, latent_dim=10, layers=1,
                          training_samples=200, pixels=16, nqubits=4, dataset=None, folder=None):
    
    # number of qubits generator
    nqubits = nqubits
    # create quantum generator
    circuit = models.Circuit(nqubits)
    for l in range(layers):
        for i in range(nqubits):
            circuit.add(gates.RY(i, 0))
        for i in range(0, nqubits - 1, 2):
            circuit.add(gates.CZ(i, i + 1))
        for i in range(nqubits):
            circuit.add(gates.RY(i, 0))
        for i in range(1, nqubits - 2, 2):
          circuit.add(gates.CZ(i, i + 1))
        circuit.add(gates.CZ(0, nqubits - 1))
    for q in range(nqubits):
        circuit.add(gates.RY(q, 0))
    # create classical discriminator
    discriminator = define_discriminator(n_inputs=pixels, lr=lr_d)
    # train model
    return train(discriminator, latent_dim, layers, nqubits, training_samples, circuit, n_epochs, batch_samples, lr, lr_d, dataset, folder)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--latent_dim", default=6, type=int)
    parser.add_argument("--layers", default=1, type=int)
    parser.add_argument("--training_samples", default=1000, type=int)
    parser.add_argument("--n_epochs", default=20000, type=int)
    parser.add_argument("--batch_samples", default=32, type=int)
    parser.add_argument("--pixels", default=16, type=int)
    parser.add_argument("--nqubits", default=4, type=int)
    parser.add_argument("--lr", default=1e-1, type=float)
    parser.add_argument("--lr_d", default=1e-2, type=float)
    parser.add_argument("--dataset", default=None, type=str)
    parser.add_argument("--folder", default=None, type=str)

    args = vars(parser.parse_args())
    build_and_train_model(**args)
