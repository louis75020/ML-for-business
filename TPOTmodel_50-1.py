import numpy as np
import pandas as pd
from sklearn.cluster import FeatureAgglomeration
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import RobustScaler
from sklearn.tree import DecisionTreeRegressor
from tpot.builtins import StackingEstimator
from xgboost import XGBRegressor
from tpot.export_utils import set_param_recursive

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'], random_state=2021)

# Average CV score on the training set was: -14.190824145869545
exported_pipeline = make_pipeline(
    FeatureAgglomeration(affinity="euclidean", linkage="complete"),
    RobustScaler(),
    StackingEstimator(estimator=GradientBoostingRegressor(alpha=0.99, learning_rate=0.1, loss="ls", max_depth=10, max_features=0.7500000000000001, min_samples_leaf=11, min_samples_split=11, n_estimators=100, subsample=0.9500000000000001)),
    StackingEstimator(estimator=DecisionTreeRegressor(max_depth=2, min_samples_leaf=7, min_samples_split=14)),
    PCA(iterated_power=2, svd_solver="randomized"),
    StackingEstimator(estimator=XGBRegressor(learning_rate=0.001, max_depth=9, min_child_weight=13, n_estimators=100, n_jobs=1, objective="reg:squarederror", subsample=0.9000000000000001, verbosity=0)),
    StackingEstimator(estimator=ElasticNetCV(l1_ratio=0.2, tol=1e-05)),
    KNeighborsRegressor(n_neighbors=2, p=1, weights="distance")
)
# Fix random state for all the steps in exported pipeline
set_param_recursive(exported_pipeline.steps, 'random_state', 2021)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
