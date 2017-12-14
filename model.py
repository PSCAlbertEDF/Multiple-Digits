import tensorflow as tf


class Model(object):

    @staticmethod
    def inference(x, drop_rate):
        with tf.variable_scope('hidden1'):
            conv = tf.layers.conv2d(x, filters=48, kernel_size=[5, 5], padding='same')
            norm = tf.layers.batch_normalization(conv)
            activation = tf.nn.relu(norm)
            pool = tf.layers.max_pooling2d(activation, pool_size=[2, 2], strides=2, padding='same')
            dropout = tf.layers.dropout(pool, rate=drop_rate)
            hidden1 = dropout

        with tf.variable_scope('hidden2'):
            conv = tf.layers.conv2d(hidden1, filters=64, kernel_size=[5, 5], padding='same')
            norm = tf.layers.batch_normalization(conv)
            activation = tf.nn.relu(norm)
            pool = tf.layers.max_pooling2d(activation, pool_size=[2, 2], strides=1, padding='same')
            dropout = tf.layers.dropout(pool, rate=drop_rate)
            hidden2 = dropout

        with tf.variable_scope('hidden3'):
            conv = tf.layers.conv2d(hidden2, filters=128, kernel_size=[5, 5], padding='same')
            norm = tf.layers.batch_normalization(conv)
            activation = tf.nn.relu(norm)
            pool = tf.layers.max_pooling2d(activation, pool_size=[2, 2], strides=2, padding='same')
            dropout = tf.layers.dropout(pool, rate=drop_rate)
            hidden3 = dropout

        with tf.variable_scope('hidden4'):
            conv = tf.layers.conv2d(hidden3, filters=160, kernel_size=[5, 5], padding='same')
            norm = tf.layers.batch_normalization(conv)
            activation = tf.nn.relu(norm)
            pool = tf.layers.max_pooling2d(activation, pool_size=[2, 2], strides=1, padding='same')
            dropout = tf.layers.dropout(pool, rate=drop_rate)
            hidden4 = dropout

        with tf.variable_scope('hidden5'):
            conv = tf.layers.conv2d(hidden4, filters=192, kernel_size=[5, 5], padding='same')
            norm = tf.layers.batch_normalization(conv)
            activation = tf.nn.relu(norm)
            pool = tf.layers.max_pooling2d(activation, pool_size=[2, 2], strides=2, padding='same')
            dropout = tf.layers.dropout(pool, rate=drop_rate)
            hidden5 = dropout

        with tf.variable_scope('hidden6'):
            conv = tf.layers.conv2d(hidden5, filters=192, kernel_size=[5, 5], padding='same')
            norm = tf.layers.batch_normalization(conv)
            activation = tf.nn.relu(norm)
            pool = tf.layers.max_pooling2d(activation, pool_size=[2, 2], strides=1, padding='same')
            dropout = tf.layers.dropout(pool, rate=drop_rate)
            hidden6 = dropout

        # with tf.variable_scope('hidden7'):
        #     conv = tf.layers.conv2d(hidden6, filters=192, kernel_size=[5, 5], padding='same')
        #     norm = tf.layers.batch_normalization(conv)
        #     activation = tf.nn.relu(norm)
        #     pool = tf.layers.max_pooling2d(activation, pool_size=[2, 2], strides=2, padding='same')
        #     dropout = tf.layers.dropout(pool, rate=drop_rate)
        #     hidden7 = dropout
        #
        # with tf.variable_scope('hidden8'):
        #     conv = tf.layers.conv2d(hidden7, filters=192, kernel_size=[5, 5], padding='same')
        #     norm = tf.layers.batch_normalization(conv)
        #     activation = tf.nn.relu(norm)
        #     pool = tf.layers.max_pooling2d(activation, pool_size=[2, 2], strides=1, padding='same')
        #     dropout = tf.layers.dropout(pool, rate=drop_rate)
        #     hidden8 = dropout

        flatten = tf.reshape(hidden6, [-1, 4 * 4 * 192])

        with tf.variable_scope('hidden9'):
            dense = tf.layers.dense(flatten, units=3072, activation=tf.nn.relu)
            hidden9 = dense

        with tf.variable_scope('hidden10'):
            dense = tf.layers.dense(hidden9, units=3072, activation=tf.nn.relu)
            hidden10 = dense

        with tf.variable_scope('digit_length'):
            dense = tf.layers.dense(hidden10, units=10)
            length = dense

        with tf.variable_scope('digit1'):
            dense = tf.layers.dense(hidden10, units=11)
            digit1 = dense

        with tf.variable_scope('letter2'):
            dense = tf.layers.dense(hidden10, units=27)
            letter2 = dense

        with tf.variable_scope('letter3'):
            dense = tf.layers.dense(hidden10, units=27)
            letter3 = dense

        with tf.variable_scope('letter4'):
            dense = tf.layers.dense(hidden10, units=27)
            letter4 = dense

        with tf.variable_scope('digit5'):
            dense = tf.layers.dense(hidden10, units=11)
            digit5 = dense

        with tf.variable_scope('digit6'):
            dense = tf.layers.dense(hidden10, units=11)
            digit6 = dense

        with tf.variable_scope('digit7'):
            dense = tf.layers.dense(hidden10, units=11)
            digit7 = dense

        with tf.variable_scope('letter8'):
            dense = tf.layers.dense(hidden10, units=27)
            letter8 = dense

        with tf.variable_scope('letter9'):
            dense = tf.layers.dense(hidden10, units=27)
            letter9 = dense

        length_logits, digits_logits, letters_logits = length, tf.stack([digit1, digit5, digit6, digit7], axis=1), \
                                                       tf.stack([letter2, letter3, letter4, letter8, letter9], axis=1)
        return length_logits, digits_logits, letters_logits

    @staticmethod
    def loss(length_logits, digits_logits, letters_logits, length_labels, digits_labels, letters_labels):
        print(length_labels)
        print(length_logits)

        length_cross_entropy = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=length_labels, logits=length_logits))
        digit1_cross_entropy = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=digits_labels[:, 0], logits=digits_logits[:, 0, :]))
        letter2_cross_entropy = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=letters_labels[:, 0], logits=letters_logits[:, 0, :]))
        letter3_cross_entropy = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=letters_labels[:, 1], logits=letters_logits[:, 1, :]))
        letter4_cross_entropy = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=letters_labels[:, 2], logits=letters_logits[:, 2, :]))
        digit5_cross_entropy = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=digits_labels[:, 1], logits=digits_logits[:, 1, :]))
        digit6_cross_entropy = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=digits_labels[:, 2], logits=digits_logits[:, 2, :]))
        digit7_cross_entropy = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=digits_labels[:, 3], logits=digits_logits[:, 3, :]))
        letter8_cross_entropy = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=letters_labels[:, 3], logits=letters_logits[:, 3, :]))
        letter9_cross_entropy = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=letters_labels[:, 4], logits=letters_logits[:, 4, :]))
        loss = length_cross_entropy + digit1_cross_entropy + letter2_cross_entropy + letter3_cross_entropy + letter4_cross_entropy + digit5_cross_entropy \
               + digit6_cross_entropy + digit7_cross_entropy + letter8_cross_entropy + letter9_cross_entropy
        return loss
