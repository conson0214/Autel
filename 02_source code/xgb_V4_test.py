# -*- coding: utf-8 -*-

import xgboost as xgb
import numpy as np


# read in data
dataset = xgb.DMatrix('xgb_train_input.txt')
