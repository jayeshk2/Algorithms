import pandas as pd
import matplotlib.pyplot as mpl
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import seaborn as sns
sns.set()

df = pd.read_csv("G:\Study\DATA SCIENCE\ExcelR\Assignments\PART-1\Simple Linear Regression\Salary_Data.csv")
print(df)

y = df.iloc[:, 1:]
x = df.iloc[:, :-1]

mpl.scatter(x, y)
mpl.xlabel("Experience", fontsize=20, color='black')
mpl.ylabel("Salary", fontsize=20, color='black')
mpl.show()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=4)

Skl_model = LinearRegression().fit(x_train, y_train)
print(Skl_model)

y_p = Skl_model.predict(x_test)
print(y_p)

score = r2_score(y_p, y_test)
print(score)
