import cv2


def split_board_image(floc, fname, out_dir, board=None):
  print(floc, fname)
  img = cv2.imread(floc)
  square_size = 150
  for r in range(0, img.shape[0], 150):
    i = r // square_size
    for c in range(0, img.shape[1], 150):
      j = c // square_size
      if board:
        piece = board[i][j]
        out_loc = f"{out_dir}/{piece}/_{fname}_{r}_{c}.jpg"
      else:
        out_loc = f"{out_dir}/{i}_{j}.jpg"

      cv2.imwrite(out_loc, img[r:r + square_size, c:c + square_size, :])
