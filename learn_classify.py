import functools
import pickle
import os
from image_features import image_features
import numpy as np
import pandas as pd
from joblib import dump, load

image_dir = 'images_chess_pieces'
# training_directories = ['front', 'back', '7']
# training_directories = ['clay', 'garrett']
training_directories = ['bR', 'bN', 'bB', 'bQ', 'bK', 'bP', 'wP', 'wR', 'wN', 'wB', 'wQ', 'wK', '__']

file_names = [(dir_name, [image_dir + '/' + dir_name + '/' + str(x) for x in os.listdir(image_dir + '/' + dir_name) if x.endswith('.jpg')]) for dir_name in training_directories]
file_names_zipped_with_label = [list(zip(front_images, [dir_name for x in front_images])) for dir_name, front_images in file_names]

df = pd.DataFrame(data=functools.reduce(lambda x,y: x+y, file_names_zipped_with_label))
df = df.rename(columns={0: "image_name", 1: "label"})
df = df.sample(frac=1).reset_index(drop=True) # shuffle rows
print(df)

if False: # if make csv
  import csv
  with open('piece_labels.csv', 'w', newline='', encoding='utf-8') as csvfile:
      csvwriter = csv.writer(csvfile, delimiter=',',
                              quotechar='|', quoting=csv.QUOTE_MINIMAL)
      # csvwriter.writerow(['set','image_path','label'])
      total_rows = len(df.index)
      print('total_rows')
      print(total_rows)
      validate = 0.2
      test = 0.0
      start_test = 1.0 - test
      start_validate = start_test - validate

      for i, row in df.iterrows():
        # TRAIN,gs://My_Bucket/sample1.jpg,cat
        percentage = i/total_rows
        set_str = 'TRAIN'
        if percentage >= start_test:
          set_str = 'TEST'
        elif percentage >= start_validate:
          set_str = 'VALIDATION'

        filename, label = row
        filename = 'gs://chess-auto-ml-vcm/' + filename
        if label == '__':
          label = 'zz'
        csvwriter.writerow([set_str, filename, label])
        # csvwriter.writerow([filename, label])

  exit()

img_paths = list(df['image_name'])
label = list(df['label'])

X_full = image_features(img_paths, progress=True)
y_full = label
try:
  print('pickling!!')
  with open('X_full.pickle', 'wb') as f:
      # Pickle the 'data' dictionary using the highest protocol available.
      pickle.dump(X_full, f, pickle.HIGHEST_PROTOCOL)
except Exception:
  print('could not pickle')


n = round(0.8 * len(img_paths))
# n = round(0.1 * len(img_paths))
X_train = X_full[:n]
y_train = y_full[:n]

X_val = X_full[n:]
y_val = y_full[n:]


from sklearn import linear_model
clf = linear_model.LogisticRegressionCV(
    max_iter=1200,
    Cs=np.geomspace(1e-1, 1e-7, 15),
    class_weight='balanced'
)
clf.fit(X_train, y_train)
print('.8 - train score:', clf.score(X_train, y_train))
print('.8 - val score:', clf.score(X_val, y_val))

print('.8 - train score:', clf.score(X_train, y_train))
print('.8 - val score:', clf.score(X_val, y_val))


clf.fit(X_full, y_full)
dump(clf, 'clf.joblib')
print('full train score:', clf.score(X_train, y_train))
print('full val score:', clf.score(X_val, y_val))

print('full full score:', clf.score(X_full, y_full))


