# -*- coding: utf-8 -*-

import os

cmd_train = """activate tensorflow && python retrain.py --architecture=inception_v4 
                                 --bottleneck_dir=V4_test/bottlenecks 
                                 --model_dir=inception_v4 
                                 --summaries_dir=V4_test/summaries/basic 
                                 --output_graph=V4_test/graph_cat_dog.pb 
                                 --output_labels=V4_test/labels_cat_dog.txt 
                                 --image_dir=datasets/dogs_vs_cats/train 
                                 --how_many_training_steps 100 
                                 --testing_percentage=5 
                                 --validation_percentage=10 
                                 --optimizer=AdamOptimizer"""

os.system(cmd_train)
