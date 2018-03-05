#!/bin/bash

python3 retrain.py --architecture=inception_v4 \
--bottleneck_dir=V4_test/bottlenecks \
--model_dir=inception_v4 \
--summaries_dir=V4_test/summaries/basic \
--output_graph=V4_test/graph_cat_dog.pb \
--output_labels=V4_test/labels_cat_dog.txt \
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
