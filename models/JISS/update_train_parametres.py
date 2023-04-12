import tensorflow as tf


def get_learning_rate(base_learning_rate, batch, decay_step, decay_rate):
    learning_rate = tf.train.exponential_decay(
        base_learning_rate,  # Base learning rate.
        batch,               # Current index into the dataset.
        decay_step,          # Decay step.
        decay_rate,          # Decay rate.
        staircase=True)
    learning_rate = tf.maximum(learning_rate, 0.00001)  # CLIP THE LEARNING RATE!!
    return learning_rate


def get_bn_decay(bn_init_decay, batch, bn_decay_decay_step, bn_decay_decay_rate, bn_decay_clip):
    bn_momentum = tf.train.exponential_decay(
        bn_init_decay,
        batch,
        bn_decay_decay_step,
        bn_decay_decay_rate,
        staircase=True)
    bn_decay = tf.minimum(bn_decay_clip, 1 - bn_momentum)
    return bn_decay