import sklearn
import numpy as np
import scipy as sp
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_boston

breastData = load_breast_cancer()
#description -> breastData.DESCR
#keys -> breastData.keys()
#numeric features -> breastData.feaure_names
#shape -> breastData.data.shape
dfFeatures = pd.DataFrame(breastData.data, columns=breastData.feature_names)
dfTarget = pd.DataFrame(breastData.target, columns=["cancer"])
df = pd.concat([dfFeatures, dfTarget], axis=1)
#print(df.head()) first five features
