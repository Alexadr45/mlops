import pandas as pd
from sklearn.model_selection import train_test_split
import gdown

#скачиваем csv файл с гугл диска и сохраняем в папке data
gdown.download(id="1waefPsrT7sm5rsRMjDHGCXreY45q9tcY", output="./data/dataset.csv", quiet=False)
#открываем данные в виде датафрейма
df = pd.read_csv('data/dataset.csv', delimiter = ',', index_col = 0)
#делим данные на тренировочные и тестовые
X_train, X_test, Y_train, Y_test = train_test_split(
    df[['id', 'year', 'code', 'period']], 
    df[['polution_clf']], 
    test_size = 0.20, 
    random_state = 42
)
#сохраняем файлы в папках train и test
X_train.to_csv('train/X_train.csv', index=True)
X_test.to_csv('test/X_test.csv', index=True)
Y_train.to_csv('train/Y_train.csv', index=False)
Y_test.to_csv('test/Y_test.csv', index=False)