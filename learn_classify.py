import functools
from functools import partial
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

def get_with_hash(obj_to_hash, cache_miss_function):
  # return cache_miss_function()
  key = str(obj_to_hash)[:8]
  filename = 'saved_objects/obj_' + key
  if os.path.exists(filename):
      print('Found', filename, 'so using that')
      with open(filename, 'rb') as f:
          return pickle.load(f)
  print('Could not find', filename, 'so manually calculating it...')
  ret = cache_miss_function()
  with open(filename, 'wb') as f:
    pickle.dump(ret, f)
  return ret


X_full = get_with_hash(len(img_paths), partial(image_features, img_paths, progress=True))
y_full = label
assert len(X_full) == len(img_paths) == len(y_full)

n = round(0.8 * len(img_paths))
# n = round(0.1 * len(img_paths))
X_train = X_full[:n]
y_train = y_full[:n]

X_val = X_full[n:]
y_val = y_full[n:]


from sklearn import linear_model, ensemble

def validate_score_clf(clf, name):
  global best_val_score
  global best_clf
  print('Fitting', name)
  try:
    clf.fit(X_train, y_train)
    print(name, '.8 - train score:', clf.score(X_train, y_train))
    val_score = clf.score(X_val, y_val)
    print(name, '.8 - val score:', val_score)
    if val_score > best_val_score:
      print('New best val score!!')
      best_clf = clf
      best_val_score = val_score
  except Exception as e:
    print("Exception!", e)
    try:
      clf.fit_transform(X_train, y_train)
      print(name, '.8 - train score:', clf.score(X_train, y_train))
      val_score = clf.score(X_val, y_val)
      print(name, '.8 - val score:', val_score)
      if val_score > best_val_score:
        print('New best val score!!')
        best_clf = clf
        best_val_score = val_score
    except Exception as e:
      print("Exception 2!", e)

best_clf = None
best_val_score = 0.95
validate_score_clf(linear_model.PassiveAggressiveClassifier(max_iter=1200), 'linear_model.PassiveAggressiveClassifier')
validate_score_clf(linear_model.SGDClassifier(max_iter=1200), 'linear_model.SGDClassifier')
validate_score_clf(linear_model.LarsCV(max_iter=1200), 'linear_model.LarsCV')
validate_score_clf(linear_model.LassoLarsCV(max_iter=1200), 'linear_model.LassoLarsCV')
validate_score_clf(linear_model.LassoCV(max_iter=1200), 'linear_model.LassoCV')
validate_score_clf(linear_model.ElasticNetCV(max_iter=1200), 'linear_model.ElasticNetCV')
validate_score_clf(linear_model.OrthogonalMatchingPursuitCV(), 'linear_model.OrthogonalMatchingPursuitCV')
validate_score_clf(ensemble.GradientBoostingClassifier(n_estimators=15, verbose=1), 'GradientBoostingClassifier')
validate_score_clf(linear_model.RidgeClassifierCV(), 'linear_model.RidgeClassifierCV')

validate_score_clf(ensemble.RandomForestClassifier(verbose=1), 'RandomForestClassifier')

clf = linear_model.LogisticRegressionCV(
    max_iter=2000, Cs=np.geomspace(1e-1, 1e-7, 15), class_weight='balanced', verbose=1
)
validate_score_clf(clf, 'LogisticRegressionCV maxiter2000')

clf = linear_model.LogisticRegressionCV(
    max_iter=900, Cs=np.geomspace(1e-1, 1e-7, 15), class_weight='balanced', verbose=1
)
validate_score_clf(clf, 'LogisticRegressionCV maxiter900')

if best_clf:
  print('Fitting best_clf with val score:', best_val_score)
  best_clf.fit(X_full, y_full)
  dump(best_clf, 'clf.joblib')

print('full train score:', clf.score(X_train, y_train))
print('full val score:', clf.score(X_val, y_val))
print('full full score:', clf.score(X_full, y_full))

from tpot import TPOTClassifier
tpot_classifier = TPOTClassifier(generations=5, population_size=20, cv=5, random_state=42, verbosity=2)
# tpot_classifier = TPOTClassifier(generations=9, population_size=20, config_dict='TPOT light', verbosity=2)
validate_score_clf(tpot_classifier, 'TPOTClassifier')
tpot_classifier.export('tpot_exported_pipeline.py')

