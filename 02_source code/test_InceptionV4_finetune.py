# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.python.platform import gfile

pb_file_path = './V4_test/graph_flowers.pb'
image_path = './5547758_eea9edfd54_n.jpg'

input_width = 299
input_height = 299
input_depth = 3
input_mean = 128
input_std = 128

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


with tf.Graph().as_default():
    output_graph_def = tf.GraphDef()

    with open(pb_file_path, "rb") as f:
        output_graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(output_graph_def, name="")

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        jpeg_data_tensor, decoded_image_tensor = add_jpeg_decoding(
            input_width, input_height,
            input_depth, input_mean,
            input_std)

        image_data = gfile.FastGFile(image_path, 'rb').read()

        input_x = sess.graph.get_tensor_by_name("InputImage:0")
        out_softmax = sess.graph.get_tensor_by_name("final_result:0")

        resized_input_values = sess.run(decoded_image_tensor,
                                        {jpeg_data_tensor: image_data})

        img_out_softmax = sess.run(out_softmax, feed_dict={input_x: resized_input_values})

    print(img_out_softmax)
