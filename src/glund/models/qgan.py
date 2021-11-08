import tensorflow as tf
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# train a quantum-classical generative adversarial network on LHC data
import numpy as np
from glund.models.optimizer import build_optimizer
from glund.models.lsgan import MinibatchDiscrimination
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Reshape, LeakyReLU, Flatten, Input
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

        self.discriminator = self.build_discriminator(units=hps['nn_units_d'],
                                                      alpha=hps['nn_alpha_d'])
        self.discriminator.compile(loss='binary_crossentropy', optimizer=self.opt, metrics=['accuracy'])


        # self.circuit = self.build_circuit(self.nqubits, self.layers)

        # params = tf.Variable(np.random.uniform(-0.15, 0.15, 5*self.layers*self.nqubits + self.nqubits))


    # discriminator
    def build_discriminator(self, units=256, alpha=0.2):
        """The GAN discriminator"""
        model = Sequential()
        model.add(Dense(units*2, input_shape=self.shape))
        model.add(LeakyReLU(alpha=alpha))
        model.add(Dense(units))
        model.add(LeakyReLU(alpha=alpha))
        model.add(Dense(1, activation='sigmoid'))
        model.summary()
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

        def generate_basis(nqubits=self.nqubits):
            basis = []
            for i in range(2**nqubits):
                test = np.zeros(2**nqubits,dtype=np.complex128)
                test[i] = 1
                basis.append(test)
            return np.array(basis)

        basis = generate_basis()
        overlaps = [callbacks.Overlap(i) for i in basis]
    
        generated_images = []
        
        for i in range(batch_size):
            self.set_params(params, circuit, x_input, i)
            circuit.execute()
            probabilities = [ i(circuit.final_state) for i in overlaps]
            normalized = 2 * (np.array(probabilities) - 0.5)
            generated_images.append(normalized)

        X = np.array(generated_images)
        y = np.zeros((batch_size, 1))

        return X, y

    def define_cost_gan(self, params, discriminator, batch_size, circuit):
        """Define the combined generator and discriminator model, for updating the generator."""
        # generate fake samples
        x_fake, y_fake = self.generate_fake_images(params, batch_size, circuit)
        # create inverted labels for the fake samples
        y_fake = np.ones((batch_size, 1))
        #print(y_fake)
        # evaluate discriminator on fake examples
        disc_output = discriminator(x_fake)
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
            self.d_loss.append(0.5 * np.add(d_loss_real, d_loss_fake))
            # ---------------------
            #  Train Generator
            # ---------------------
            with tf.GradientTape() as tape:
                loss = self.define_cost_gan(initial_params, self.discriminator, batch_size, circuit)
                #print(tape.watched_variables())
            # print(initial_params)
            grads = tape.gradient(loss, initial_params)
            print(grads)
            self.opt.apply_gradients([(grads, initial_params)])

            self.g_loss.append(loss)
           
            
            if epoch%10==0:
                print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]"
                       % (epoch, self.d_loss[0], 100*self.d_loss[1], self.g_loss))

            
    # -------------------------------------------------
    def generate(self, circuit,nev):
        
        _, images = self.generate_fake_images(nev, circuit)
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

        np.savetxt(f"PARAMS_LundJet_{self.nqubits}_{self.latent_dim}_{self.layers}"%folder, [params.numpy()], newline='')
        np.savetxt(f"dloss_LundJet_{self.nqubits}_{self.latent_dim}_{self.layers}"%folder, [self.d_loss], newline='')
        np.savetxt(f"gloss_LundJet_{self.nqubits}_{self.latent_dim}_{self.layers}"%folder, [self.g_loss], newline='')
        self.discriminator.save_weights(f"discriminator_LundJet_{self.nqubits}_{self.atent_dim}_{self.layers}.h5"%folder)

    #------------------------------------------------------------------------
    def description(self):
        return 'QGAN with length=%i, latent_dim=%i' % (self.length, self.latent_dim)
