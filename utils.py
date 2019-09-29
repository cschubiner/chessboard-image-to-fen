import cv2


def split_board_image(floc, fname, out_dir, board=None):
  img = cv2.imread(floc)
  square_size = 150
  for r in range(0, img.shape[0], 150):
    i = r // square_size
    for c in range(0, img.shape[1], 150):
      j = c // square_size
      if board:
        piece = board[i][j]
        out_loc = str(out_dir) + "/" + str(piece) + "/_" + str(fname) + "_" + str(r) + "_" + str(c) + ".jpg"
      else:
        out_loc = str(out_dir) + "/" + str(i) + "_" + str(j) + ".jpg"

      cv2.imwrite(out_loc, img[r:r + square_size, c:c + square_size, :])
