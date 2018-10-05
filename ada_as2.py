# coding: utf-8

# In[2]:


get_ipython().run_line_magic('config', 'IPCompleter.greedy=True')
import warnings

warnings.filterwarnings("ignore")
import datetime
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import preprocessing
from sklearn.dummy import DummyClassifier
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, recall_score, accuracy_score, precision_score
import itertools
from mpl_toolkits.basemap import Basemap

get_ipython().run_line_magic('matplotlib', 'inline')
import tensorflow as tf
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC

print('OK')

# In[3]:


terror = pd.read_csv("d:\\Work\\ADA\\gtdb_2000.csv", low_memory=False, encoding='ISO-8859-1')
terror['casualties'] = terror['nkill'] + terror['nwound']
terror.target1 = terror.target1.str.lower()
terror.gname = terror.gname.str.lower()
terror.head(2)

# In[5]:


terror['cas_bin'] = terror['casualties'].apply(
    lambda x: 0 if x == 0 else 1)  # shortcut if else logic to provide true false
gtd_pred = ['iyear',
            'imonth',
            'iday',
            'latitude',
            'longitude',
            'attacktype1',
            'region',
            'success',
            'weaptype1',
            'target1',
            'INT_LOG',
            'INT_IDEO',
            'gname',
            'property',
            'targtype1',
            'country']
target_pred = 'cas_bin'

lb = preprocessing.LabelEncoder()
terror['gname'] = lb.fit_transform(terror['gname'])
terror['target1'] = lb.fit_transform(terror['target1'])
x = terror[gtd_pred].fillna(0)
y = terror[target_pred]

forest = ExtraTreesClassifier(n_estimators=250, random_state=0)
forest.fit(x, y)
importance = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
indices = np.argsort(importance)[::-1]
frames = [gtd_pred[i] for i in indices]

print('done')

# In[6]:


plt.figure(figsize=(10, 6), dpi=220)
plt.bar(range(x.shape[1]), importance[indices], color="rgby", yerr=std[indices], align="center")
plt.xticks(range(x.shape[1]), frames, rotation=90)
plt.title("Prediction of casualty occurance: feature importances")
plt.xlim([-1, x.shape[1]])
plt.grid(which='major', linestyle=':', linewidth='0.5', color='red')
plt.grid(which='minor', linestyle='-', linewidth='0.5', color='black')
plt.show()

# In[7]:


gtdb_pred_col = ['attacktype1', 'targtype1', 'target1', 'country', 'iday', 'imonth', 'iyear', 'longitude', 'latitude']
X = terror[gtdb_pred_col].fillna(0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)
model = RandomForestClassifier(n_estimators=150)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
np.mean(y_pred == y_test)

# In[8]:


cmatrix = confusion_matrix(y_test, y_pred)


def plot_confusion_matrix(cm, normalize=False, title='Confusion matrix', cmap=plt.cm.Paired):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Original label')
    plt.xlabel('Predicted label')


np.set_printoptions(precision=2)

plt.figure(figsize=(10, 6), dpi=220)
plot_confusion_matrix(cmatrix,
                      title='Confusion matrix - No Normalisation')
plt.figure(figsize=(10, 6), dpi=220)
plot_confusion_matrix(cmatrix, normalize=True,
                      title='Confusion Matrix - Normalised')
plt.show()

# In[9]:


print('Recall score: %0.2f' % recall_score(y_test, y_pred))
print('Accuracy score: %0.2f' % accuracy_score(y_test, y_pred))
print('Precision score: %0.2f' % precision_score(y_test, y_pred))

# In[10]:


model = RandomForestClassifier(n_estimators=64)

scores = cross_val_score(model, X_train, y_train, cv=20)
print('Accuracy: %0.2f (+/- %0.2f)' % (scores.mean(), scores.std() * 2))

# In[28]:


model = DummyClassifier(strategy="most_frequent")

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

np.mean(y_pred == y_test)

# In[54]:


from sklearn.model_selection import GridSearchCV, cross_val_score, cross_val_predict
from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(
    hidden_layer_sizes=(10),
    activation='logistic',
    learning_rate='adaptive',
    learning_rate_init=0.3,
    random_state=56,
    max_iter=500,
    solver='sgd'
)

mlp_grid = GridSearchCV(mlp, mlp_params, scoring='accuracy', cv=5, n_jobs=-1, error_score=0)

mlp_grid.fit(X_train, y_train)

y1_pred = mlp_grid.predict(X_test)

print(np.mean(y1_pred == y_test))

print(mlp_grid.score(X_test, y_test))
print(mlp_grid.best_params_)

