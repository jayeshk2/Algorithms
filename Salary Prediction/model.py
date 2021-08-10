import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import pickle

df = pd.read_csv('G:\Technologies\DATA SCIENCE\ExcelR\Assignments\PART-1\Simple Linear Regression\Salary_Data.csv')

y = df.iloc[:, 1:]
x = df.iloc[:, :-1]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=4)

Regression_Model = LinearRegression()
Regression_Model.fit(x_train, y_train)
y_pred = Regression_Model.predict(x_test)


Score = r2_score(y_test, y_pred)
print(Score)

pickle.dump(Regression_Model, open('Regression_Model.pkl', 'wb'))
