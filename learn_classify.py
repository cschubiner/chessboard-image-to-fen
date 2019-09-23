import functools
import os
from image_features import image_features
import numpy as np
import pandas as pd
from joblib import dump, load

image_dir = 'images_chess_pieces'
# training_directories = ['front', 'back', '7']
# training_directories = ['clay', 'garrett']
training_directories = ['bR', 'bN', 'bB', 'bQ', 'bK', 'bP', 'wP', 'wR', 'wN', 'wB', 'wQ', 'wK', '--']

file_names = [(dir_name, [image_dir + '/' + dir_name + '/' + str(x) for x in os.listdir(image_dir + '/' + dir_name) if x.endswith('.jpg')]) for dir_name in training_directories]
file_names_zipped_with_label = [list(zip(front_images, [dir_name for x in front_images])) for dir_name, front_images in file_names]

df = pd.DataFrame(data=functools.reduce(lambda x,y: x+y, file_names_zipped_with_label))
df = df.rename(columns={0: "image_name", 1: "label"})
df = df.sample(frac=1).reset_index(drop=True) # shuffle rows
print(df)

img_paths = list(df['image_name'])
label = list(df['label'])

X_full = image_features(img_paths, progress=True)
y_full = label

n = round(0.8 * len(img_paths))
# n = round(0.1 * len(img_paths))
X_train = X_full[:n]
y_train = y_full[:n]

X_val = X_full[n:]
y_val = y_full[n:]


from sklearn import linear_model
clf = linear_model.LogisticRegressionCV(
    max_iter=2000,
    Cs=np.geomspace(1e-1, 1e-7, 15),
    class_weight='balanced'
)
# clf.fit(X_train, y_train)
clf.fit(X_full, y_full)
print('train score:', clf.score(X_train, y_train))
print('val score:', clf.score(X_val, y_val))

dump(clf, 'clf.joblib')
print('train score:', clf.score(X_train, y_train))
print('val score:', clf.score(X_val, y_val))

