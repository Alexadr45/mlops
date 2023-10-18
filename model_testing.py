import pandas as pd
import pickle
from sklearn.metrics import f1_score
import json

LogReg = pickle.load(open('model.pkl', 'rb'))
X_test = pd.read_csv('data/test/X_test.csv', delimiter = ',')
Y_test = pd.read_csv('data/test/Y_test.csv', delimiter = ',')
y_preds = LogReg.predict(X_test)
score = f1_score(Y_test, y_preds, average="micro")

with open('evaluate/score.json', 'w') as f:
    json.dump({"score": score}, f)
