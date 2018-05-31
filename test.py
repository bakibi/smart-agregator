import tensorflow as tf
import numpy as np
import pandas as pd


graph = tf.Graph()
with graph.as_default():
    sess = tf.Session(graph=graph)

    x = tf.placeholder(shape=[None], dtype=tf.float32)
    y = tf.placeholder(shape=[None],dtype=tf.float32)
    y = x

    print(sess.run(y,{x : [1]}))
