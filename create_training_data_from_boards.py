from labeled_boards import labeled_boards
import cv2

def chunks(l, n):
  for i in range(0, len(l), n):
    yield l[i:i+n]

out_dir = 'images_chess_pieces'


def split_board_image():
  global i, c
  img = cv2.imread('clayboards_out/' + fname)
  square_size = 150
  for r in range(0, img.shape[0], 150):
    i = r // square_size
    for c in range(0, img.shape[1], 150):
      j = c // square_size
      piece = board[i][j]
      cv2.imwrite(f"{out_dir}/{piece}/_{fname}_{r}_{c}.jpg", img[r:r + square_size, c:c + square_size, :])


for fname, board_str in labeled_boards.items():
  if not fname or not board_str or len(board_str) < 10:
    continue

  board = list()
  board_lines = board_str.split('\n')
  for i, line in enumerate(board_lines):
    if not line and (i == 0 or i == len(board_lines) - 1):
      continue
    if not line:
      board.append(['--'] * 8)
    else:
      row = []
      for c in chunks(line, 2):
        c = c[0].lower() + c[1].upper()
        # print(fname, c, len(board))
        assert c in ['bR', 'bN', 'bB', 'bQ', 'bK', 'bP', 'wP', 'wR', 'wN', 'wB', 'wQ', 'wK', '--']
        row.append(c)
      assert len(row) == 8
      board.append(row)
  # print(board)
  assert len(board) == 8

  # Now need to split up the image and put it into folders
  split_board_image(board)

