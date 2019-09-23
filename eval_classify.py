import functools
import os
import numpy as np
import pandas as pd
from utils import split_board_image
from image_features import image_features
from joblib import dump, load

board_image_dir = 'to_eval'
board_file_names = [str(x) for x in os.listdir(board_image_dir)]

pieces_image_dir = 'to_eval_pieces'

clf = load('clf.joblib')

for fname in board_file_names:
  print(f"Evaluating {fname}...")
  split_board_image(board_image_dir + '/' + fname, fname, pieces_image_dir)

  pieces_file_names = [str(x) for x in os.listdir(pieces_image_dir)]
  pieces_file_locs = [pieces_image_dir + '/' + x for x in pieces_file_names]
  pieces_image_features = image_features(pieces_file_locs)

  pieces_preds = clf.predict(pieces_image_features)

  print(len(pieces_file_names), len(pieces_file_locs), len(pieces_image_features), len(pieces_preds))
  assert len(pieces_file_names) == len(pieces_file_locs) == len(pieces_image_features) == len(pieces_preds)
  assert 64 == len(pieces_file_names)

  pred_board_dict = dict()
  for index, fname in enumerate(pieces_file_names):
    i, j, key = fname[0], fname[2], fname[:3]
    pred = pieces_preds[index]
    pred_board_dict[key] = pred

  for i in range(8):
    for j in range(8):
      key = f"{i}_{j}"
      print(pred_board_dict[key], end = ' ')
    print('')
