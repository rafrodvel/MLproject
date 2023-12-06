import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import sklearn.preprocessing as skl_pre
import sklearn.linear_model as skl_lm
import sklearn.discriminant_analysis as skl_da
import sklearn.neighbors as skl_nb
import sklearn.model_selection as skl_ms

#from IPython.display import set_matplotlib_formats
#set_matplotlib_formats('png')
from IPython.core.pylabtools import figsize
figsize(10, 6) # Width and hight
#plt.style.use('seaborn-white')

data = pd.read_csv('training_data.csv')
datanorm = pd.read_csv('normalized_data.csv')

X = datanorm.drop('increase_stock', axis=1)
y = datanorm['increase_stock']

model = skl_lm.LogisticRegression(solver='newton-cg')
model.fit(X, y)
print(f'Logistic regression train error using all dataset is {np.mean(model.predict(X) != y)}')

# skl.ms.tain_test_split

model = skl_lm.LogisticRegression(solver='newton-cg')

n_folds = 15
kf = skl_ms.KFold(n_splits=n_folds + 1, random_state=2, shuffle=True)

lst_index = []
missclasification = 0
missclasification2 = []

for train_index, val_index in kf.split(X):
    lst_index.append((train_index, val_index))
    if len(lst_index) <= n_folds:
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        error_rate = np.mean(y_pred != y_val)
        missclasification += error_rate
        missclasification2.append(error_rate)

average_error_rate = missclasification / n_folds
print(f'Average error rate across {n_folds} folds: {average_error_rate}')

# Now, let's use the model to predict on the left aside dataset
left_aside_data = datanorm.iloc[lst_index[-1][1]]
X_left_aside = left_aside_data.drop('increase_stock', axis=1)
y_left_aside = left_aside_data['increase_stock']

model.fit(X, y)
y_pred_left_aside = model.predict(X_left_aside)
left_aside_error_rate = np.mean(y_pred_left_aside != y_left_aside)
print(f'Error rate on the left aside dataset: {left_aside_error_rate}')

boxplot = plt.boxplot(missclasification2)
plt.title(f'Boxplot of the error rate across the {n_folds} folds with logistic regression')
plt.ylabel('Error rate')
plt.xlabel('Logistic regression')
plt.show()








