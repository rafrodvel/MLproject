import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import sklearn.preprocessing as skl_pre
import sklearn.linear_model as skl_lm
import sklearn.discriminant_analysis as skl_da
import sklearn.neighbors as skl_nb
from sklearn.model_selection import train_test_split, KFold

#from IPython.display import set_matplotlib_formats
#set_matplotlib_formats('png')
from IPython.core.pylabtools import figsize
figsize(10, 6) # Width and hight
#plt.style.use('seaborn-white')

data = pd.read_csv('training_data.csv')
datanorm = pd.read_csv('normalized_data.csv')

X = datanorm.drop('increase_stock', axis=1)
y = datanorm['increase_stock']

model = skl_lm.LogisticRegression(solver='liblinear')
model.fit(X, y)
print(f'Logistic regression train error using all dataset is {np.mean(model.predict(X) != y)}')

# skl.ms.tain_test_split

model = skl_lm.LogisticRegression(solver='liblinear')

from sklearn.model_selection import train_test_split, KFold
import numpy as np
import matplotlib.pyplot as plt

# Split the data into training and test sets (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

# Initialize KFold
n_folds = 16
kf = KFold(n_splits=n_folds)

missclasification = 0
missclasification2 = []
lst_index = []

for train_index, val_index in kf.split(X_train):
    lst_index.append((train_index, val_index))
    X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
    y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]
    
    model.fit(X_train_fold, y_train_fold)
    y_pred = model.predict(X_val_fold)
    error_rate = np.mean(y_pred != y_val_fold)
    missclasification += error_rate
    missclasification2.append(error_rate)

average_error_rate = missclasification / n_folds
print(f'Average error rate across {n_folds} folds: {average_error_rate}')

# Now, let's use the model to predict on the test dataset
model.fit(X_train, y_train)
y_pred_test = model.predict(X_test)
test_error_rate = np.mean(y_pred_test != y_test)
print(f'Error rate on the test dataset: {test_error_rate}')

boxplot = plt.boxplot(missclasification2)
plt.title(f'Boxplot of the error rate across the {n_folds} folds with logistic regression')
plt.ylabel('Error rate')
plt.xlabel('Logistic regression')

solvers = ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
max_iter = 10000

for solver in solvers:
    model = skl_lm.LogisticRegression(solver=solver, max_iter=max_iter)
    model.fit(X_train, y_train)
    print(f'Logistic regression train error using {solver} solver is {np.mean(model.predict(X_test) != y_test)}')





