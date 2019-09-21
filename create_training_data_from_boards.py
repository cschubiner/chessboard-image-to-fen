from labeled_boards import labeled_boards

out_dir = 'images_test/'

for fname, board_str in labeled_boards.items():
  if not fname or not board_str or len(board_str) < 10:
    continue

  board = list()
  board_lines = board_str.split('\n')
  for i, line in enumerate(board_lines):
    if not line and (i == 0 or i == len(board_lines) - 1):
      continue
    if not line:
      board.append(['-'] * 8)
    else:
      row = []
      for c in line:
        row.append(c)
      assert len(row) == 8
      board.append(row)
  print(board)
  assert len(board) == 8

  # Now need to split up the image and put it into folders
  import cv2

  img = cv2.imread('clayboards_out/' + fname)
  for r in range(0,img.shape[0],150):
      for c in range(0,img.shape[1],150):
          cv2.imwrite(f"{fname}_{r}_{c}.jpg",img[r:r+150, c:c+150,:])

