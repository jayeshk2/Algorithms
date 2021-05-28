import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as mpl
import seaborn as sns
sns.set()

df = pd.read_csv("G:\Study\DATA SCIENCE\ExcelR\Assignments\PART-1\Simple Linear Regression\calories_consumed.csv")
print(df)

y = df.iloc[:, :-1]
x = df.iloc[:, 1:]

print(y)


mpl.scatter(x, y)
mpl.xlabel('Calories', fontsize=20, color='red')
mpl.ylabel('Weight Gain', fontsize=20, color='red')
mpl.show()


abc = df.isnull().sum()
print(abc)

xyz = df.duplicated().sum()
print(xyz)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=4)

Skl_Model = LinearRegression().fit(x_train, y_train)

y_p = Skl_Model.predict(x_test)
print(y_p)

Score = r2_score(y_test, y_p)
print(Score)
