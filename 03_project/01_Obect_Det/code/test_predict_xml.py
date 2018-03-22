# coding=utf-8

import tensorflow as tf
import numpy as np
import xml.etree.ElementTree as ET
import cv2
import sys
import os

def preprocess_func(input_width, input_height, input_depth, input_mean,
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
    # jpeg_data = tf.placeholder(tf.string, name='DecodeJPGInput')
    # decoded_image = tf.image.decode_jpeg(jpeg_data, channels=input_depth)
    img_data = tf.placeholder(tf.uint8, [None, None, input_depth])
    decoded_image_as_float = tf.cast(img_data, dtype=tf.float32)
    decoded_image_4d = tf.expand_dims(decoded_image_as_float, 0)
    resize_shape = tf.stack([input_height, input_width])
    resize_shape_as_int = tf.cast(resize_shape, dtype=tf.int32)
    resized_image = tf.image.resize_bilinear(decoded_image_4d,
                                             resize_shape_as_int)
    offset_image = tf.subtract(resized_image, input_mean)
    mul_image = tf.multiply(offset_image, 1.0 / input_std)
    return img_data, mul_image


def get_label_name(label_fle):
    label_names = []
    with open(label_fle, 'r') as label_file:
        while 1:
            line = label_file.readline()
            if not line:
                break
            line = line.replace('\n', '')
            label_names.append(line)
    return label_names


def read_xml(xml_path):
    with open(xml_path) as xml_file:
        tree = ET.parse(xml_file)
    return tree

def write_xml(tree, out_path):
  tree.write(out_path, encoding="utf-8", xml_declaration=True)


def getallfile(path, target):
    allfile = []
    for dirpath, dirnames, filenames in os.walk(path):
        for name in filenames:
            if target in filenames:
                allfile.append(os.path.join(dirpath, name))
    return allfile


def main(img_name, test_foler, pb_file_name, label_file_name):
    model_info = {}
    model_info['input_width'] = 224
    model_info['input_height'] = 224
    model_info['input_depth'] = 3
    model_info['input_mean'] = 127.5
    model_info['input_std'] = 127.5
    pb_file_path = pb_file_name

    label_list = get_label_name(label_file_name)
    tree = read_xml(xml_in_name)
    root = tree.getroot()
    raw_img = cv2.imread(img_name)
    b, g, r = cv2.split(raw_img)
    raw_img = cv2.merge([r, g, b])

    with tf.Graph().as_default():
        output_graph_def = tf.GraphDef()

        with open(pb_file_path, "rb") as f:
            output_graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(output_graph_def, name="")

        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)

            img_data_tensor, decoded_image_tensor = preprocess_func(
                model_info['input_width'], model_info['input_height'],
                model_info['input_depth'], model_info['input_mean'],
                model_info['input_std'])

            input_x = sess.graph.get_tensor_by_name("input_1:0")
            out_softmax = sess.graph.get_tensor_by_name("final_result:0")

            for obj in root.iter('object'):
                xmlbox = obj.find('bndbox')
                x_min = int(xmlbox.find('xmin').text)
                x_max = int(xmlbox.find('xmax').text)
                y_min = int(xmlbox.find('ymin').text)
                y_max = int(xmlbox.find('ymax').text)
                b = (x_min, x_max, y_min, y_max)
                bb_img = raw_img[b[2]:b[3], b[0]:b[1]]

                resized_input_values = sess.run(decoded_image_tensor, {img_data_tensor: bb_img})
                img_out_softmax = sess.run(out_softmax, feed_dict={input_x: resized_input_values})
                predict = int(np.argmax(img_out_softmax))
                predict_class = label_list[predict]
                obj.find('name').text = predict_class
    write_xml(tree, xml_out_name)


img_name = sys.argv[1]
xml_in_name = sys.argv[2]
xml_out_name = sys.argv[3]
if sys.argv.__len__() <= 4:
    pb_file_name = 'graph_class6_nasnet.pb'
else:
    pb_file_name = sys.argv[4]
if sys.argv.__len__() <= 5:
    label_file_name = 'labels_list.txt'
else:
    label_file_name = sys.argv[5]

# img_name = 'test.jpg'
# xml_in_name = 'test.xml'
# xml_out_name = 'result.xml'
# pb_file_name = 'graph_class6_nasnet.pb'
# label_file_name = 'labels_list.txt'

main(img_name, xml_in_name, xml_out_name, pb_file_name, label_file_name)
