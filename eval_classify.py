import functools
import os
import numpy as np
import pandas as pd
from utils import split_board_image
from image_features import image_features
from joblib import dump, load

board_image_dir = 'to_eval'
default_board_file_names = [board_image_dir + '/' + str(x) for x in os.listdir(board_image_dir)]

pieces_image_dir = 'to_eval_pieces'

clf = load('clf.joblib')

def eval_images(board_file_names=None):
  if not board_file_names:
    board_file_names = default_board_file_names

  for fname in board_file_names:
    print("Evaluating " + str(fname) + "...")
    split_board_image(fname, fname, pieces_image_dir)

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

    ret = list()
    for i in range(8):
      row = ''
      for j in range(8):
        key = str(i) + "_" + str(j)
        row += pred_board_dict[key] + ' '
      ret.append(row.strip())
    return ret

if __name__ == "__main__":
  result = eval_images()
  print('\n'.join(result))

