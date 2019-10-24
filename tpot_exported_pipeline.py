import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

# NOTE: Make sure that the class is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1).values
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'].values, random_state=42)

# Average CV score on the training set was:0.9447711936498349
exported_pipeline = LinearSVC(C=0.01, dual=True, loss="hinge", penalty="l2", tol=0.1)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
