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

CLAY_TO_FEN_DICT = {
  'bN': 'n',
  'bB': 'b',
  'bQ': 'q',
  'bK': 'k',
  'bP': 'p',
  'bR': 'r',
  'wN': 'N',
  'wB': 'B',
  'wQ': 'Q',
  'wK': 'K',
  'wP': 'P',
  'wR': 'R',
}

def board_to_fen(board):
  fen = ''
  for i in range(8):
      j = 0
      while j < 8:
          claypiece = board[i][j]
          if claypiece == '--':
            count = 1
            for k in range(j + 1, 8):
              if board[i][k] == '--':
                count += 1
              else:
                break
            fen += str(count)
            print(k)
            j += count
            continue
          fen += CLAY_TO_FEN_DICT[claypiece]
          j += 1
      fen += '/'
  fen = fen[:-1]
  fen += ' w KQkq - 0 1'
  return fen

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

    board = list()
    for i in range(8):
      row = list()
      for j in range(8):
        key = str(i) + "_" + str(j)
        row.append(pred_board_dict[key])
      board.append(row)


    return board, board_to_fen(board)

if __name__ == "__main__":
  # result, fen = eval_images()
  print(board_to_fen([['wB', 'bN', '--', 'bK', '--', '--', '--', '--'], ['bP', '--', 'bP', '--', '--', '--', 'bP', 'bP'], ['--', '--', '--', 'bP', '--', '--', '--', '--'], ['--', '--', '--', '--', '--', '--', '--', '--'], ['--', '--', 'wK', '--', '--', 'wN', 'wP', '--'], ['--', '--', '--', 'wP', 'wB', 'wB', 'wP', '--'], ['wP', 'wP', 'wP', 'wN', 'wR', '--', '--', '--'], ['wR', '--', '--', '--', '--', 'bN', '--', '--']]))
  # print('\n'.join(result))

