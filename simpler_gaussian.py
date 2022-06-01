import tensorflow as tf
import numpy as np
from numpy.random import randn
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adadelta
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Reshape, LeakyReLU, Flatten
from qibo import gates, models, set_backend
import argparse


set_backend('tensorflow')

# define the standalone discriminator model
def define_discriminator(n_inputs=64, alpha=0.2, dropout=0.2, lr=0.1):
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

def generate_training_real_samples(samples):
    from sklearn.datasets import load_digits
    img_data = load_digits(n_class=1).images
    # Rescale 0 to 1
    img_data = img_data.astype(np.float32) / 16.
    img_data = img_data[:samples]
    return np.reshape(img_data, (img_data.shape[0], 64))
 
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
        circuit_execute = circuit.execute()
        max_val = max(abs(circuit.final_state.numpy())**2)
        for ii in range(pixels):
                X[ii].append(abs(circuit.final_state[ii]**2) / max_val)

    # shape array
    X = tf.stack([X[i] for i in range(len(X))], axis=1)
    # create class labels
    y = np.zeros((samples, 1))
    return X, y

# train the generator and discriminator
def train(d_model, latent_dim, layers, nqubits, training_samples, discriminator, circuit, n_epochs, samples, lr, pixels, lr_d):
    d_loss = []
    g_loss = []
    # determine half the size of one batch, for updating the discriminator
    half_samples = int(samples / 2)
    initial_params = tf.Variable(np.random.uniform(0, 2*np.pi, 4*layers*nqubits + 2*nqubits))
    optimizer = tf.optimizers.Adadelta(learning_rate=lr)
    # prepare real samples
    s = generate_training_real_samples(training_samples)
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
        #print("grads", grads)
        optimizer.apply_gradients([(grads, initial_params)])
        g_loss.append(loss)
        np.savetxt(f"PARAMS_Handwritten-0-digit_{nqubits}_{latent_dim}_{layers}_{training_samples}_{samples}_{lr}_{lr_d}", [initial_params.numpy()], newline='')
        np.savetxt(f"dloss_Handwritten-0-digit_{nqubits}_{latent_dim}_{layers}_{training_samples}_{samples}_{lr}_{lr_d}", [d_loss], newline='')
        np.savetxt(f"gloss_Handwritten-0-digit_{nqubits}_{latent_dim}_{layers}_{training_samples}_{samples}_{lr}_{lr_d}", [g_loss], newline='')
        # serialize weights to HDF5
        #discriminator.save_weights(f"discriminator_Handwritten-0-digit_{nqubits}_{latent_dim}_{layers}_{training_samples}_{samples}_{lr}_{lr_d}.h5")
    return loss

def build_and_train_model(lr_d=1e-2, lr=1e-2, n_epochs=10, batch_samples=10, latent_dim=3, layers=1, training_samples=100, pixels=64, nqubits=6):
    
    # number of qubits generator
    nqubits = nqubits
    # create quantum generator
    circuit = models.Circuit(nqubits)
    for l in range(layers):
        for q in range(nqubits):
            circuit.add(gates.RY(q, 0))
            circuit.add(gates.RY(q, 0))
        for i in range(0, nqubits-1):
            circuit.add(gates.CZ(i, i+1))
        circuit.add(gates.CZ(nqubits-1, 0))
    for q in range(nqubits):
        circuit.add(gates.RY(q, 0))
    # create classical discriminator
    discriminator = define_discriminator(n_inputs=pixels, lr=lr_d)
    # train model
    return train(discriminator, latent_dim, layers, nqubits, training_samples, discriminator, circuit, n_epochs, batch_samples, lr, pixels, lr_d)
