import tensorflow as tf
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# train a quantum-classical generative adversarial network on LHC data
import numpy as np
from glund.models.optimizer import build_optimizer
from tensorflow.keras.models import Sequential, Model 
from tensorflow.keras.layers import Dense, LeakyReLU, Conv2D, Dropout, ZeroPadding2D, Flatten, BatchNormalization, Input
from qibo import gates, hamiltonians, models, set_backend, set_threads

set_backend('tensorflow')

# helper functions

def hamiltonian1():
    id = [[1, 0], [0, 1]]
    m0 = hamiltonians.Z(1).matrix  # no numpy=True argument
    m0 = np.kron(id, m0)
    ham = hamiltonians.Hamiltonian(2, m0)
    return ham
    
def hamiltonian2():
    id = [[1, 0], [0, 1]]
    m0 = hamiltonians.Z(1).matrix # no numpy= True argument
    m0 = np.kron(m0, id)
    ham = hamiltonians.Hamiltonian(2, m0)
    return ham

def define_cost_gan(params, discriminator, latent_dim, samples, circuit, nqubits, layers, hamiltonian1, hamiltonian2):
    # generate fake samples
    x_fake, y_fake = generate_fake_images(params, latent_dim, samples, circuit, nqubits, layers, hamiltonian1, hamiltonian2)
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
            p.append(params[index]*x_input[noise][i] + params[index+1])
            index+=2
            noise=(noise+1)%latent_dim
            p.append(params[index]*x_input[noise][i] + params[index+1])
            index+=2
            noise=(noise+1)%latent_dim
        if l==1 or l==5 or l==9 or l==13 or l==17:
            p.append(params[index]*x_input[noise][i] + params[index+1])
            index+=2
            noise=(noise+1)%latent_dim
        if l==3 or l==7 or l==11 or l==15 or l==19:
            p.append(params[index]*x_input[noise][i] + params[index+1])
            index+=2
            noise=(noise+1)%latent_dim
    for q in range(nqubits):
        p.append(params[index]*x_input[noise][i] + params[index+1])
        index+=2
        noise=(noise+1)%latent_dim
    circuit.set_parameters(p)
 
# generate real images 
def generate_real_images(X_train, batch_size):
    # generate samples from the distribution
    idx = np.random.randint(0, X_train.shape[0], batch_size)
    imgs = X_train[idx]
    # generate class labels
    y = np.ones((batch_size, 1))
    return imgs, y

# generate points in latent space as input for the generator
def generate_noise(latent_dim, batch_size):
    noise = np.random.normal(0, 1, (batch_size, latent_dim))
    return noise

# use the generator to generate fake examples, with class labels
def generate_fake_images(circuit, params, latent_dim, batch_size, nqubits, layers, hamiltonian1, hamiltonian2):
    # generate points in latent space
    x_input = generate_noise(latent_dim, batch_size)
    x_input = np.transpose(x_input)
    # generator outputs
    X1 = []
    X2 = []
    # quantum generator circuit
    for i in range(batch_size):
        set_params(circuit, params, x_input, i, nqubits, layers, latent_dim)
        circuit_execute = circuit.execute()
        X1.append(hamiltonian1.expectation(circuit_execute))
        X2.append(hamiltonian2.expectation(circuit_execute))
    # shape array
    X = tf.stack((X1, X2), axis=1)
    # create class labels
    y = np.zeros((batch_size, 1))
    return X, y

class QGAN():

    #------------------------------------------------------------------------
    def __init__(self, hps, layers, nqubits, length=28*28,):
        self.length = length
        self.shape  = (self.length,)
        self.latent_dim = hps['latdim']
        self.layers = layers
        self.nqubits = nqubits

        opt = build_optimizer(hps)

        self.discriminator = self.build_discriminator(units=hps['nn_units_d'],
                                                      alpha=hps['nn_alpha'],
                                                      momentum=hps['nn_momentum_d'],
                                                      dropout=hps['nn_dropout'])

        self.discriminator.compile(loss='binary_crossentropy',
                                   optimizer=opt,
                                   metrics=['accuracy'])

        self.circuit = self.build_circuit(nqubits, layers)

        self.params = tf.Variable(np.random.uniform(-0.15, 0.15, 4*self.layers*self.nqubits + 2*self.nqubits + 2*self.layers))

    # discriminator in dcgan
    def build_discriminator(self, units=32, alpha=0.2, momentum=0.8, dropout=0.25):

        model = Sequential()

        model.add(Conv2D(units, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same"))
        model.add(LeakyReLU(alpha=alpha))
        model.add(Dropout(dropout))
        model.add(Conv2D(units*2, kernel_size=3, strides=2, padding="same"))
        model.add(ZeroPadding2D(padding=((0,1),(0,1))))
        model.add(BatchNormalization(momentum=momentum))
        model.add(LeakyReLU(alpha=alpha))
        model.add(Dropout(dropout))
        model.add(Conv2D(units*4, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=momentum))
        model.add(LeakyReLU(alpha=alpha))
        model.add(Dropout(dropout))
        model.add(Conv2D(units*8, kernel_size=3, strides=1, padding="same"))
        model.add(BatchNormalization(momentum=momentum))
        model.add(LeakyReLU(alpha=alpha))
        model.add(Dropout(dropout))
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))

        model.summary()

        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)

    def build_circuit(self):
        circuit = models.Circuit(self.nqubits)
        for l in range(self.layers):
            for q in range(self.nqubits):
                circuit.add(gates.RY(q, 0))
                circuit.add(gates.RZ(q, 0))
            if l==1 or l==5 or l==9 or l==13 or l==17:
                circuit.add(gates.CRY(0, 1, 0))
            if l==3 or l==7 or l==11 or l==15 or l==19:
                circuit.add(gates.CRY(1, 0, 0))
        for q in range(self.nqubits):
            circuit.add(gates.RY(q, 0))

        return circuit

    def train(self,X_train, epochs, batch_size=128):

        self.d_losses = []
        self. g_losses = []

        # manually enumerate epochs
        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # prepare real samples
            x_real, y_real = generate_real_images(X_train, batch_size)
            # prepare fake examples
            x_fake, y_fake = generate_fake_images(self.circuit, self.params, self.latent_dim, batch_size, self.nqubits, self.layers, hamiltonian1, hamiltonian2)
            # update discriminator
            d_loss_real = self.discriminator.train_on_batch(x_real, y_real)
            d_loss_fake = self.discriminator.train_on_batch(x_fake, y_fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            self.d_losses.append((d_loss_real + d_loss_fake)/2)
            # ---------------------
            #  Train Generator
            # ---------------------

            with tf.GradientTape() as tape:
                loss = define_cost_gan(self.params, self.discriminator, self.latent_dim, batch_size,
                                       self.circuit, self.nqubits, self.layers, hamiltonian1, hamiltonian2)
                grads = tape.gradient(loss, self.params)

            self.opt.apply_gradients([(grads, self.params)])
            self.g_losses = loss
           
            
            if epoch%10==0:
                print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]"
                       % (epoch, self.d_loss[0], 100*self.d_loss[1], self.g_loss))

            
    # -------------------------------------------------
    def generate(self, nev):
        
        noise, images = generate_fake_images(self.circuit, self.params, self.latent_dim, nev,
                                             self.nqubits, self.layers, hamiltonian1, hamiltonian2)
        return images

    #-----------------------------------------------------
    def load(self, folder):
        """Load qGAN from input folder"""
        # load the weights from input folder
        # self.generator.load_weights('%s/generator.h5'%folder)
        # self.discriminator.load_weights('%s/discriminator.h5'%folder)
        pass

    #----------------------------------------------------------------------
    def save(self, folder):
        """Save the qGAN weights to file."""
        # self.discriminator.save_weights('%s/discriminator.h5'%folder)
        # personalized save file

        np.savetxt(f"PARAMS_LundJet_{self.nqubits}_{self.latent_dim}_{self.layers}"%folder, [self.params.numpy()], newline='')
        np.savetxt(f"dloss_LundJet_{self.nqubits}_{self.latent_dim}_{self.layers}"%folder, [self.d_loss], newline='')
        np.savetxt(f"gloss_LundJet_{self.nqubits}_{self.latent_dim}_{self.layers}"%folder, [self.g_loss], newline='')
        self.discriminator.save_weights(f"discriminator_LundJet_{self.nqubits}_{self.atent_dim}_{layers}.h5"%folder)

    #------------------------------------------------------------------------
    def description(self):
        return 'QGAN with length=%i, latent_dim=%i' % (self.length, self.latent_dim)


if __name__ == '__main__':
