# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.python.platform import gfile
import numpy as np
import os
import pandas as pd

architecture = "Mobilenet"

if architecture == "Inception-V3":
    project_path = './V3_test/'
    pb_file_name = 'graph_cat_dog_V3.pb'
elif architecture == "Inception-V4":
    project_path = './V4_test/'
    pb_file_name = 'graph_cat_dog.pb'
elif architecture == "Mobilenet":
    project_path = './Mobilenet_test/'
    pb_file_name = 'graph_cat_dog_mobilenet.pb'
else:
    tf.logging.error("Couldn't understand architecture name '%s'", architecture)
    raise ValueError('Unknown architecture', architecture)

image_path = './datasets/dogs_vs_cats/test/'
submission_name = 'submission.csv'

model_info = {}
if architecture == "Inception-V3":
    model_info['input_width'] = 299
    model_info['input_height'] = 299
    model_info['input_depth'] = 3
    model_info['input_mean'] = 128
    model_info['input_std'] = 128
elif architecture == "Inception-V4":
    model_info['input_width'] = 299
    model_info['input_height'] = 299
    model_info['input_depth'] = 3
    model_info['input_mean'] = 128
    model_info['input_std'] = 128
elif architecture == "Mobilenet":
    model_info['input_width'] = 224
    model_info['input_height'] = 224
    model_info['input_depth'] = 3
    model_info['input_mean'] = 127.5
    model_info['input_std'] = 127.5
else:
    tf.logging.error("Couldn't understand architecture name '%s'", architecture)
    raise ValueError('Unknown architecture', architecture)


extensions = ['.jpg', '.jpeg', '.JPG', '.JPEG']

def add_jpeg_decoding(input_width, input_height, input_depth, input_mean,
                      input_std):
  """Adds operations that perform JPEG decoding and resizing to the graph..

  Args:
    input_width: Desired width of the image fed into the recognizer graph.
    input_height: Desired width of the image fed into the recognizer graph.
    input_depth: Desired channels of the image fed into the recognizer g    raph.
    input_mean: Pixel value that should be zero in the image for the graph.
    input_std: How much to divide the pixel values by before recognition.

  Returns:
    Tensors for the node to feed JPEG data into, and the output of the
      preprocessing steps.
  """
  jpeg_data = tf.placeholder(tf.string, name='DecodeJPGInput')
  decoded_image = tf.image.decode_jpeg(jpeg_data, channels=input_depth)
  decoded_image_as_float = tf.cast(decoded_image, dtype=tf.float32)
  decoded_image_4d = tf.expand_dims(decoded_image_as_float, 0)
  resize_shape = tf.stack([input_height, input_width])
  resize_shape_as_int = tf.cast(resize_shape, dtype=tf.int32)
  resized_image = tf.image.resize_bilinear(decoded_image_4d,
                                           resize_shape_as_int)
  offset_image = tf.subtract(resized_image, input_mean)
  mul_image = tf.multiply(offset_image, 1.0 / input_std)
  return jpeg_data, mul_image

pb_file_path = os.path.join(project_path, pb_file_name)
submission_path = os.path.join(project_path, submission_name)
image_list = os.listdir(image_path)
index_list = list(range(1, 12501))
label_list = [0]*12500
process_idx = 0

with tf.Graph().as_default():
    output_graph_def = tf.GraphDef()

    with open(pb_file_path, "rb") as f:
        output_graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(output_graph_def, name="")

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        jpeg_data_tensor, decoded_image_tensor = add_jpeg_decoding(
            model_info['input_width'], model_info['input_height'],
            model_info['input_depth'], model_info['input_mean'],
            model_info['input_std'])

        for f in image_list:
            (name, extension) = os.path.splitext(f)
            if extension in extensions:
                process_idx += 1
                print("process %dth image" % process_idx)
                image_index = int(name) - 1
                image_whole_path = os.path.join(image_path, f)

                image_data = gfile.FastGFile(image_whole_path, 'rb').read()

                if architecture == "Inception-V3":
                    input_x = sess.graph.get_tensor_by_name("Mul:0")  # InceptionV3
                elif architecture == "Inception-V4":
                    input_x = sess.graph.get_tensor_by_name("InputImage:0")  # InceptionV4
                elif architecture == "Mobilenet":
                    input_x = sess.graph.get_tensor_by_name("input:0")  # Mobilenet
                else:
                    tf.logging.error("Couldn't understand architecture name '%s'", architecture)
                    raise ValueError('Unknown architecture', architecture)

                out_softmax = sess.graph.get_tensor_by_name("final_result:0")
                resized_input_values = sess.run(decoded_image_tensor,
                                                {jpeg_data_tensor: image_data})
                img_out_softmax = sess.run(out_softmax, feed_dict={input_x: resized_input_values})
                label_list[image_index] = 1 - np.clip(img_out_softmax[0][0], 0.05, 0.95)
                # predict = np.argmax(img_out_softmax)
                #
                # label_list[image_index] = predict

dataframe = pd.DataFrame({'id': index_list, 'label': label_list})
dataframe.to_csv(submission_path, index=False, sep=',')
