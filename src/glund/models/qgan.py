import tensorflow as tf
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# train a quantum-classical generative adversarial network on LHC data
import numpy as np
from glund.models.optimizer import build_optimizer
from glund.models.lsgan import MinibatchDiscrimination
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Reshape, LeakyReLU, Flatten, Input
from scipy.optimize import minimize
from qibo import gates, models, set_backend, callbacks

set_backend('tensorflow')

class QGAN():

    #------------------------------------------------------------------------
    def __init__(self, hps, length=28*28):
        self.length = length
        self.shape  = (self.length,)
        self.latent_dim = hps['latdim']
        self.layers = hps['layers']
        self.nqubits = hps['nqubits']

        self.opt = build_optimizer(hps)

        self.discriminator = self.build_discriminator(dropout=hps['nn_dropout'],
                                                      alpha=hps['nn_alpha_d'])
        self.discriminator.compile(loss='binary_crossentropy', optimizer=self.opt, metrics=['accuracy'])


    # discriminator
    def build_discriminator(self, alpha=0.2, dropout=0.2):
        model = Sequential()
        
        model.add(Dense(200, use_bias=False, input_dim=int(2**self.nqubits)))
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

        return model

    def generate_latent_points(self, batch_size):
        """Generate points in latent space as input for the quantum generator."""
        # generate points in the latent space
        x_input = np.random.randn(self.latent_dim * batch_size)
        # reshape into a batch of inputs for the network
        x_input = x_input.reshape(batch_size, self.latent_dim)
        return x_input

    # use the generator to generate fake examples, with class labels
    def generate_fake_images(self, params, batch_size, circuit):
        # generate points in latent space
        x_input = self.generate_latent_points(batch_size)
        x_input = np.transpose(x_input)

        pxl = int(2**self.nqubits)

        # generator outputs
        X = []
        for i in range(pxl):
            X.append([])

        for i in range(batch_size):
            self.set_params(params, circuit, x_input, i)
            circuit_execute = circuit.execute()
            max_val = max(abs(circuit.final_state.numpy()))
            for ii in range(pxl):
                X[ii].append(abs(circuit.final_state[ii])/max_val)

        X = tf.stack([X[i] for i in range(len(X))], axis=1)
        y = np.zeros((batch_size, 1))

        return X, y

    def define_cost_gan(self, params, discriminator, batch_size, circuit):
        """Define the combined generator and discriminator model, for updating the generator."""
        # generate fake samples
        x_fake, y_fake = self.generate_fake_images(params, batch_size, circuit)
        # create inverted labels for the fake samples
        y_fake = np.ones((batch_size, 1))
        #print(x_fake.shape)
        # evaluate discriminator on fake examples
        disc_output = discriminator(x_fake)
        #print(disc_output)
        #print(y_fake, disc_output)
        loss = tf.keras.losses.binary_crossentropy(y_fake, disc_output)
        loss = tf.reduce_mean(loss)

        return loss

    def set_params(self, params, circuit, x_input, i):
        """Set the parameters for the quantum generator circuit."""
        p = []
        index = 0
        noise = 0
        for l in range(self.layers):
            for q in range(self.nqubits):
                p.append(params[index]*x_input[noise][i] + params[index+1])
                index+=2
                noise=(noise+1)%self.latent_dim
                p.append(params[index]*x_input[noise][i] + params[index+1])
                index+=2
                p.append(params[index]*x_input[noise][i] + params[index+1])
                index+=2
                noise=(noise+1)%self.latent_dim
                p.append(params[index]*x_input[noise][i] + params[index+1])
                index+=2
                noise=(noise+1)%self.latent_dim
            for i in range(0, self.nqubits-1):
                p.append(params[index]*x_input[noise][i] + params[index+1])
                index+=2
                noise=(noise+1)%self.latent_dim
            p.append(params[index]*x_input[noise][i] + params[index+1])
            index+=2
            noise=(noise+1)%self.latent_dim
        for q in range(self.nqubits):
            p.append(params[index]*x_input[noise][i] + params[index+1])
            index+=2
            noise=(noise+1)%self.latent_dim
        circuit.set_parameters(p)

    def train(self,X_train, epochs, batch_size=128):

        self.batch_size = batch_size
        # generate real images 
        def generate_real_images(X_train, batch_size):
            # generate samples from the distribution
            idx = np.random.randint(X_train.shape[0], size=batch_size)
            imgs = X_train[idx]
            # generate class labels
            y = np.ones((batch_size, 1))
            return imgs, y

        self.d_loss = []
        self.g_loss = []

        initial_params = tf.Variable(np.random.uniform(0, 2*np.pi, 10*self.layers*self.nqubits + 2*self.nqubits))
        half_batch_size = int(batch_size / 2)

        # create quantum generator
        circuit = models.Circuit(self.nqubits)
        for l in range(self.layers):
            for q in range(self.nqubits):
                circuit.add(gates.RY(q, 0))
                circuit.add(gates.RZ(q, 0))
                circuit.add(gates.RY(q, 0))
                circuit.add(gates.RZ(q, 0))
            for i in range(0, self.nqubits-1):
                circuit.add(gates.CRY(i, i+1, 0))
            circuit.add(gates.CRY(self.nqubits-1, 0, 0))
        for q in range(self.nqubits):
            circuit.add(gates.RY(q, 0))

        # manually enumerate epochs
        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------
            # prepare real samples
            x_real, y_real = generate_real_images(X_train, half_batch_size)
            # prepare fake examples
            x_fake, y_fake = self.generate_fake_images(initial_params, half_batch_size, circuit)
            # update discriminator
            d_loss_real = self.discriminator.train_on_batch(x_real, y_real)
            d_loss_fake = self.discriminator.train_on_batch(x_fake, y_fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            # ---------------------
            #  Train Generator
            # ---------------------
            with tf.GradientTape() as tape:
                loss = self.define_cost_gan(initial_params, self.discriminator, self.batch_size, circuit)

            grads = tape.gradient(loss, initial_params)
            print(grads)
            self.opt.apply_gradients([(grads, initial_params)])

            np.savetxt(f"PARAMS_digits_{self.nqubits}_{self.latent_dim}_{self.layers}", [initial_params.numpy()], newline='')
            np.savetxt(f"dloss_digits_{self.nqubits}_{self.latent_dim}_{self.layers}", [d_loss], newline='')
            np.savetxt(f"gloss_digits_{self.nqubits}_{self.latent_dim}_{self.layers}", [loss], newline='')

            self.g_loss.append(loss)
           
            
            if epoch%10==0:
                print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]"
                       % (epoch, d_loss[0], 100*d_loss[1], loss))

        # after the training we store circuit and params for generate method
        self.params = initial_params
        self.circuit = circuit

            
    # -------------------------------------------------
    def generate(self, nev):
        
        images, _ = self.generate_fake_images(self.params, nev, self.circuit)
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

        #np.savetxt(f"PARAMS_LundJet_{self.nqubits}_{self.latent_dim}_{self.layers}"%folder, [params.numpy()], newline='')
        #np.savetxt(f"dloss_LundJet_{self.nqubits}_{self.latent_dim}_{self.layers}"%folder, [self.d_loss], newline='')
        #np.savetxt(f"gloss_LundJet_{self.nqubits}_{self.latent_dim}_{self.layers}"%folder, [self.g_loss], newline='')
        #self.discriminator.save_weights(f"discriminator_LundJet_{self.nqubits}_{self.atent_dim}_{self.layers}.h5"%folder)
        pass

    #------------------------------------------------------------------------
    def description(self):
        return 'QGAN with length=%i, latent_dim=%i' % (self.length, self.latent_dim)
