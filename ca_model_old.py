import tensorflow as tf
from tensorflow.keras.layers import Conv2D
import numpy as np

class CAModel(tf.keras.Model):
    
    def __init__(self, NO_CHANNELS, NO_CLASSES, H, W, add_noise=True, full_model=True):
        super().__init__()
        self.add_noise = add_noise
        self.NO_CHANNELS = NO_CHANNELS
        self.NO_CLASSES = NO_CLASSES
        self.H = H
        self.W = W

        if full_model:
            self.update_state = tf.keras.Sequential([
                Conv2D(80, 3, activation=tf.nn.relu, padding="SAME"),
                Conv2D(120, 1, activation=tf.nn.relu, padding="SAME"),
                Conv2D(NO_CHANNELS, 1, activation=None, padding="SAME"),
            ])
        else:
            self.update_state = tf.keras.Sequential([
                Conv2D(NO_CHANNELS, 3, activation=None, padding="SAME"),
            ])
        
        self(tf.zeros([1, H, W, 1 + NO_CHANNELS])) # dummy call to build the model
    
    @tf.function
    def call(self, x):
        '''
        this function updates the CA for one cycle
        x is the current CA state. its shape is (batch_size, H, W, no_channels). 
            batch_size is BATCH_SIZE.
            no_channels is 1 + NO_CHANNELS, 
                where the first is the gray image, 
                the last NO_CLASSES are the classification predictions,
                and the others are there just for fun :)
        '''
        new_state = self.update_state(x) # new_state will be the state update (of course, we don't want to update the gray image as that is our true input)
        image, old_state = tf.split(x, [1, self.NO_CHANNELS], -1)
        if self.add_noise:
            residual_noise = tf.random.normal(tf.shape(new_state), mean=0., stddev=0.02)
            new_state += residual_noise

        new_state *= image 

        return tf.concat([image, new_state], -1)

    @tf.function
    def initialize(self, images):
        '''
        input: images of size (batch, h, w)
        output: initial CA state full of 0's for positions other than the images. shape (batch, h, w, 1 + channel_n)
        '''
        state = tf.zeros([tf.shape(images)[0], self.H, self.W, self.NO_CHANNELS]) # size (batch, h, w, channel_n) full of zeros
        images = tf.reshape(images, [-1, self.H, self.W, 1]) # our images we add an extra dimension
        return tf.concat([images, state], -1) # just concatenating

    @tf.function
    def initialize_random(self, images):
        '''
        input: images of size (batch, h, w)
        output: initial CA state full of 0's for positions other than the images. shape (batch, h, w, 1 + channel_n)
        '''
        state = tf.random.normal([tf.shape(images)[0], self.H, self.W, self.NO_CHANNELS]) # size (batch, h, w, channel_n) with random numbers
        images = tf.reshape(images, [-1, self.H, self.W, 1]) # our images we add an extra dimension
        return tf.concat([images, state], -1) # just concatenating

    @tf.function
    def classify(self, x):
        '''
        The last NO_CLASSES layers are the classification predictions, one channel
        per class.
        '''
        return x[:, :, :, -self.NO_CLASSES:]
    
    @tf.function
    def mutate(self, x, new_images):
        '''
        This function corrupts the current state of the CA by just changing the gray image
        '''
        old_images, state = tf.split(x, [1, self.NO_CHANNELS], -1)
        return tf.concat([new_images, state], -1)




# Training utilities
@tf.function
def individual_l2_loss(ca, x, y):
    '''
    x is the current CA state vector. its shape is (batch_size, height, width, no_channels).
    y is the correct label out of NO_CLASSES possibilities. its shape is (batch_size, height, width, NO_CLASSES) (one-hot)
    '''
    t = y - ca.classify(x) # basically we want 1's for the correct and 0s for the incorrect digit. its shape is (batch_size, height, width, NO_CLASSES) (one-hot)
    error_batch = tf.reduce_sum(t ** 2, [1, 2, 3]) / 2
    no_pixels = tf.reduce_sum(y, [1, 2, 3])
    error_normalised_batch = error_batch / no_pixels
    return error_normalised_batch

@tf.function
def batch_l2_loss(ca, x, y, label_vector, NO_CLASSES):
    '''
    x is the current CA state vector. its shape is (batch_size, height, width, no_channels).
    y is the correct label out of 10 possibilities. its shape is (batch_size, height, width, 10) (one-hot)
    returns the mean of the loss function
    '''
    i_l = individual_l2_loss(ca, x, y)
    class_loss = []
    for i in range(NO_CLASSES):
        idx = tf.where((label_vector > (i - 0.5)) & (label_vector < (i + 0.5)))
        gather = tf.gather(i_l, tf.reshape(idx, (-1,)))
        class_loss.append(tf.reduce_mean(gather))
    return tf.reduce_mean(i_l), class_loss


def export_model(folder, id_run, ca, i, loss_log, loss_log_classes):
    '''
    Saves the models parameters
    '''
    ca.save_weights(folder + '/saved_models/' + id_run + '_run_no_{}'.format(i))
    np.savez(folder + '/saved_models/' + id_run + '_run_no_{}_loss'.format(i), loss_log=loss_log, loss_log_classes=loss_log_classes)

def get_model(folder, id_run, i, NO_CHANNELS, NO_CLASSES, H, W, ADD_NOISE):
    '''
    Gets the models parameters
    '''
    ca = CAModel(NO_CHANNELS, NO_CLASSES, H, W, add_noise=ADD_NOISE)
    ca.load_weights(folder + '/saved_models/' + id_run + '_run_no_{}'.format(i))
    res = np.load(folder + '/saved_models/' + id_run + '_run_no_{}_loss.npz'.format(i))
    loss_log = res['loss_log']
    loss_log_classes = res['loss_log_classes']
    return ca, loss_log, loss_log_classes


@tf.function
def train_step(trainer, ca, x, y, y_label, TR_EVOLVE, NO_CLASSES, MutateTrainingQ=False):
    '''
    x is the current CA state. its shape is (batch_size, height, width, no_channels).
    y is the correct label out of 10 possibilities. its shape is (batch_size, ?)
    '''
    iter_n = max(2, np.random.randint(TR_EVOLVE)) # Number of initial iterations of the CA for each training step
    with tf.GradientTape() as g: # GradientTape does automatic differentiation on the learnable_parameters of our model
        for i_iter in tf.range(iter_n): # Basically let time evolve
            x = ca(x) # update the CA according to call method? ca(x) = ca.call(x)?
        loss_b, c_l_b = batch_l2_loss(ca, x, y, y_label, NO_CLASSES) # compute the scalar loss
    grads = g.gradient(loss_b, ca.weights) # Gradient Tape and Keras doing its magic
    grads = [g/(tf.norm(g)+1e-8) for g in grads] # Normalising the gradients uh?
    trainer.apply_gradients(zip(grads, ca.weights)) # Keras and ADAM magic 
    

    if MutateTrainingQ:
        indices = tf.range(start=0, limit=tf.shape(x)[0], dtype=tf.int32)
        shuffled_indices = tf.random.shuffle(indices)

        shuffled_images = tf.gather(x[:, :, :, 0], shuffled_indices)
        x = ca.mutate(x, tf.expand_dims(shuffled_images, -1))
        shuffled_y = tf.gather(y, shuffled_indices)
        shuffled_y_label = tf.gather(y_label, shuffled_indices)
    else:
        shuffled_y = y
        shuffled_y_label = y_label
    
    iter_n = TR_EVOLVE - iter_n # Number of iterations of the CA for each training step
    with tf.GradientTape() as g: # GradientTape does automatic differentiation on the learnable_parameters of our model
        for i_iter in tf.range(iter_n): # Basically let time evolve
            x = ca(x) # update the CA according to call method? ca(x) = ca.call(x)?
        loss_a, c_l_a = batch_l2_loss(ca, x, shuffled_y, shuffled_y_label, NO_CLASSES) # compute the scalar loss
    grads = g.gradient(loss_a, ca.weights) # Gradient Tape and Keras doing its magic
    grads = [g/(tf.norm(g)+1e-8) for g in grads] # Normalising the gradients uh?
    trainer.apply_gradients(zip(grads, ca.weights)) # Keras and ADAM magic 
    return x, loss_b + loss_a, [c_l_b[i_list] + c_l_a[i_list] for i_list in range(NO_CLASSES)]