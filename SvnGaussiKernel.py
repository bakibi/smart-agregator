import tensorflow as tf




'''
    Cette classe decrit les differentes tranformations de la methodes gaussien
'''


class SvnGaussiKernel(object) :
    def __init__(self,MAX_SEQUENCE,ALL_X,ALL_Y,BATCH_SIZE=4,NUMBER_OF_CATEGORIES=7):
        self.x_data = tf.placeholder(shape=[None, MAX_SEQUENCE], dtype=tf.float32)
        self.y_target = tf.placeholder(shape=[NUMBER_OF_CATEGORIES, None], dtype=tf.float32)
        self.prediction_grid = tf.placeholder(shape=[None, MAX_SEQUENCE], dtype=tf.float32)

        # Create variables for svm
        b = tf.Variable(tf.random_normal(shape=[NUMBER_OF_CATEGORIES, BATCH_SIZE]))

        with tf.device('/cpu:0'), tf.name_scope('SVM'):
            # Gaussian (RBF) kernel
            gamma = tf.constant(-15.0)
            dist = tf.reduce_sum(tf.square(self.x_data), 1)
            dist = tf.reshape(dist, [-1, 1])
            sq_dists = tf.multiply(2., tf.matmul(self.x_data, tf.transpose(self.x_data)))
            my_kernel = tf.exp(tf.multiply(gamma, tf.abs(sq_dists)))

            # Declare function to do reshape/batch multiplication
            def reshape_matmul(mat, _size):
                v1 = tf.expand_dims(mat, 1)
                v2 = tf.reshape(v1, [NUMBER_OF_CATEGORIES, _size, 1])
                return tf.matmul(v2, v1)

            # Compute SVM Model
            first_term = tf.reduce_sum(b)
            b_vec_cross = tf.matmul(tf.transpose(b), b)
            y_target_cross = reshape_matmul(self.y_target,   BATCH_SIZE)
            second_term = tf.reduce_sum(tf.multiply(my_kernel, tf.multiply(b_vec_cross, y_target_cross)), [1, 2])
            self.loss = tf.reduce_sum(tf.negative(tf.subtract(first_term, second_term)))

            # Gaussian (RBF) prediction kernel
            rA = tf.reshape(tf.reduce_sum(tf.square(self.x_data), 1), [-1, 1])
            rB = tf.reshape(tf.reduce_sum(tf.square(self.prediction_grid), 1), [-1, 1])
            pred_sq_dist = tf.add(tf.subtract(rA, tf.multiply(2., tf.matmul(self.x_data, tf.transpose(self.prediction_grid)))), tf.transpose(rB))
            pred_kernel = tf.exp(tf.multiply(gamma, tf.abs(pred_sq_dist)))

            prediction_output = tf.matmul(tf.multiply(self.y_target, b), pred_kernel)
            self.prediction = tf.argmax(prediction_output - tf.expand_dims(tf.reduce_mean(prediction_output, 1), 1), 0)
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.prediction, tf.argmax(self.y_target, 0)), tf.float32))

            # Declare optimizer
            my_opt = tf.train.GradientDescentOptimizer(0.098)
            self.train_step = my_opt.minimize(self.loss)
