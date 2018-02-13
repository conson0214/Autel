#!/bin/bash

python3 retrain.py --architecture=inception_v3 \
--bottleneck_dir=V3_test/bottlenecks \
--model_dir=inception_v3 \
--summaries_dir=V3_test/summaries/basic \
--output_graph=V3_test/graph_cat_dog_V3.pb \
--output_labels=V3_test/labels_cat_dog.txt \
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
