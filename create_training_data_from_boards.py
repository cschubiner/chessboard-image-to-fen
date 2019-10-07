from utils import split_board_image
from labeled_boards import labeled_boards


def chunks(l, n):
  for i in range(0, len(l), n):
    yield l[i:i+n]


if __name__ == "__main__":

  for fname, board_str in labeled_boards.items():
    if not fname or not board_str or len(board_str) < 10:
      continue

    board = list()
    board_lines = board_str.split('\n')
    for i, line in enumerate(board_lines):
      if not line and (i == 0 or i == len(board_lines) - 1):
        continue
      if not line:
        board.append(['__'] * 8)
      else:
        row = []
        for c in chunks(line, 2):
          c = c[0].lower() + c[1].upper()
          # print(fname, c, len(board))
          assert c in ['bR', 'bN', 'bB', 'bQ', 'bK', 'bP', 'wP', 'wR', 'wN', 'wB', 'wQ', 'wK', '__']
          row.append(c)
        assert len(row) == 8
        board.append(row)
    # print(board)
    assert len(board) == 8

    # Now need to split up the image and put it into folders
    # clayboards_out is input folder
    import os.path
    fpath = 'clayboards_out/' + fname
    if os.path.exists(fpath):
      split_board_image(fpath, fname, 'images_chess_pieces', board)

