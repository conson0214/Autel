# -*- coding: utf-8 -*-

import tensorflow as tf

output_graph_path = r"./V4_test/graph_cat_dog.pb"
with tf.Session() as sess:
    # with tf.gfile.FastGFile(output_graph_path, 'rb') as f:
    #     graph_def = tf.GraphDef()
    #     graph_def.ParseFromString(f.read())
    #     sess.graph.as_default()
    #     tf.import_graph_def(graph_def, name='')
    tf.global_variables_initializer().run()
    output_graph_def = tf.GraphDef()
    with open(output_graph_path, "rb") as f:
        output_graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(output_graph_def, name="")

train_writer = tf.summary.FileWriter("./train/", sess.graph)