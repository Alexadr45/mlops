#создайте python-скрипт , который создает и обучает модель машинного обучения на построенных данных из папки “train”.
from sklearn.linear_model import LogisticRegression
import pandas as pd
import pickle

X_train = pd.read_csv('train/X_train.csv', delimiter = ',')
y_train = pd.read_csv('train/Y_train.csv', delimiter = ',')
model = LogisticRegression(fit_intercept=True,
                            penalty='l1',solver='liblinear',
                            C=8.75,
                            max_iter=10000)
model.fit(X_train, y_train)

pickle.dump(model, open('model.pkl', 'wb'))