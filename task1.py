import pandas as pd
import matplotlib.pylab as plt
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error #measures the distance between 
# your h which are prediction by our model and actual test values 
from sklearn.metrics import r2_score #how much variation in exam numbers is 
#explained by study hours 
df = pd.read_csv('StudentPerformanceFactors.csv')
df.plot.scatter(x = 'Hours_Studied' , y ='Exam_Score')
plt.show()
x = df[['Hours_Studied']]
y = df['Exam_Score']
x_train , x_test, y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)
model = LinearRegression()
model.fit(x_train, y_train)
h = model.predict(x_test)
print(h)
s = mean_squared_error(h,y_test)
print(s)
g = r2_score(y_test,h)
print(g)
plt.scatter(x_test,y_test, color = 'blue', label = 'ActualLine')
plt.scatter(x_test,h,color = 'Red', label = 'predictedLine')
plt.show()
# Check for missing values in your features and target
print(df[['Hours_Studied', 'Exam_Score']].isnull().sum())

