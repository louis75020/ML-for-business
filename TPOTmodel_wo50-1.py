import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline, make_union
from tpot.builtins import StackingEstimator
from tpot.export_utils import set_param_recursive

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'], random_state=2021)

# Average CV score on the training set was: -35.717505054555915
exported_pipeline = make_pipeline(
    VarianceThreshold(threshold=0.01),
    StackingEstimator(estimator=RandomForestRegressor(bootstrap=False, max_features=0.5, min_samples_leaf=1, min_samples_split=2, n_estimators=100)),
    StackingEstimator(estimator=RandomForestRegressor(bootstrap=False, max_features=0.45, min_samples_leaf=1, min_samples_split=4, n_estimators=100)),
    KNeighborsRegressor(n_neighbors=15, p=1, weights="distance")
)
# Fix random state for all the steps in exported pipeline
set_param_recursive(exported_pipeline.steps, 'random_state', 2021)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
