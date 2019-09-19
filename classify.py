import functools
import os
from image_features import image_features
import numpy as np
import pandas as pd

training_directories = ['front', 'back', '7']
file_names = [(dir_name, ['images/' + dir_name + '/' + str(x) for x in os.listdir('images/' + dir_name)]) for dir_name in training_directories]
front_images = ['images/front/' + str(x) for x in os.listdir('images/front')]
back_images = ['images/back/' + str(x) for x in os.listdir('images/back')]

file_names_zipped_with_label = [list(zip(front_images, [dir_name for x in front_images])) for dir_name, front_images in file_names]
ret = functools.reduce(lambda x,y: x+y, file_names_zipped_with_label)
print(ret)

front_x = zip(front_images, [True for x in front_images])
back_x = zip(back_images, [False for x in back_images])
d = list(front_x) + list(back_x)
# print(d)

df = pd.DataFrame(data=ret)
print(df)
df = df.rename(columns={0: "image_name", 1: "is_front"})
print(df)


img_paths = list(df['image_name'])
is_front = list(df['is_front'])

n = round(0.8 * len(img_paths))
X_train = image_features(img_paths[:n], progress=True)
y_train = is_front[:n]

X_val = image_features(img_paths[n:], progress=True)
y_val = is_front[n:]


from sklearn import linear_model
clf = linear_model.LogisticRegressionCV(
    max_iter=1000,
    Cs=np.geomspace(1e-1, 1e-7, 15),
    class_weight='balanced'
)
clf.fit(X_train, y_train)
print('train score:', clf.score(X_train, y_train))
print('val score:', clf.score(X_val, y_val))
