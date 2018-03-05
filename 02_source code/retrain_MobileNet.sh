#!/bin/bash

IMAGE_SIZE=224
ARCHITECTURE="mobilenet_0.50_${IMAGE_SIZE}"

python3 retrain.py --architecture="${ARCHITECTURE}" \
--bottleneck_dir=Mobilenet_test/bottlenecks \
--model_dir="${ARCHITECTURE}" \
--summaries_dir=Mobilenet_test/summaries/basic \
--output_graph=Mobilenet_test/graph_cat_dog_mobilenet.pb \
--output_labels=Mobilenet_test/labels_cat_dog.txt \
--image_dir=datasets/dogs_vs_cats/train \
--how_many_training_steps 10000 \
--testing_percentage=5 \
--validation_percentage=10 \
--optimizer=Adam \
--flip_left_right \
--random_crop=5 \
--random_scale=5 \
--random_brightness=5 \
--learning_rate=0.001
