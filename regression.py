# Simple Linear Regression

# Importing the libraries
import numpy
import matplotlib.pyplot as plot
import pandas

# Importing the dataset
dataset = pandas.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values # Took all the rows(:) of all the cols except last (:-1)
y = dataset.iloc[:, 1].values # Dependent variables, took all rows(:) of last col(3)

# Splitting the dataset into Training set and Test set
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=0)

# Feature Scaling
## (put all values on the same scale)
'''
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
'''

# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting Test set
y_pred = regressor.predict(X_test)

# Visualizing the Training set results
plot.scatter(X_train, y_train, color='red')
plot.plot(X_train, regressor.predict(X_train), color='blue')
plot.title('Salary vs Experience(Training set)')
plot.xlabel('Years of Experience')
plot.ylabel('Salary')
plot.show()

print(regressor.score(X_train, regressor.predict(X_train)))

# Visualizing the Test set results
plot.scatter(X_test, y_test, color='red')
plot.plot(X_test, y_pred, color='blue')
plot.title('Salary vs Experience(Test set)')
plot.xlabel('Years of Experience')
plot.ylabel('Salary')
plot.show()

print(regressor.score(X_test, y_test))
