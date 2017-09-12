from statistics import mean
import numpy as np
import matplotlib.pyplot as plt

xs = [1, 2,3,4, 5, 6]
ys = [5,4,6,5,6,7]

plt.scatter(xs,ys)
plt.show()

#find the best fit slope
xs = np.array([1, 2,3,4, 5, 6], dtype = np.float64)
ys = np.array([5,4,6,5,6,7], dtype = np.float64)

def best_fit_slope(xs,ys):
    m = ( (mean(xs)* mean(ys)) - mean(xs*ys))/ ((mean(xs)**2) - (mean(xs**2)))
    return m 

m = best_fit_slope(xs,ys)
print(m)

#O/P  - 0.428571428571

def best_fit_slope_and_intercept(xs,ys):
    m = ( (mean(xs)* mean(ys)) - mean(xs*ys))/ ((mean(xs)**2) - (mean(xs**2)))
    b = mean(ys) - m*mean(xs)
    return m, b

m, b = best_fit_slope_and_intercept(xs,ys)
print(m,b)

#O/P - 0.428571428571 4.0

#Checking the regression line in visualisation 

from matplotlib import style
style.use('fivethirtyeight')
regression_line = [(m*x)+b for x in xs]
plt.scatter(xs,ys)
plt.plot(xs, regression_line)
plt.show()

#Programming the squared error / coefficient of determination which is kind of looked upon as the accuracy of the model fit

def squared_error(ys_orig, ys_line):
    return sum((ys_line-ys_orig)** 2)

def coefficient_of_determination(ys_orig, ys_line):
    y_mean_line = [mean(ys_orig) for y in ys_orig]
    squared_error_reg = squared_error(ys_orig, ys_line)
    squared_error_y_mean = squared_error(ys_orig, y_mean_line)
    return 1 - ((squared_error_reg )/squared_error_y_mean)

r_squared = coefficient_of_determination(ys, regression_line)
print(r_squared)

#O/P -0.584415584416

#Testing for assumptions
#Ignore the hardcore dataset defined above for xs and ys and create a dataset
#hm - how many datapoints you wish to have
import random 
def create_dataset(hm, variance, step=2, correlation = False ):
    val = 1
    ys = []
    for i in range(hm):
        y = val +random.randrange(-variance, variance)
        ys.append(y)
        if correlation and correlation == 'pos':
            val +=step
        elif correlation and correlation == 'neg':
            val -=step
    xs = [i for i in range(len(ys))]
    return np.array(xs, dtype=np.float64), np.array(ys, dtype = np.float64)

xs, ys = create_dataset(40,8,2, correlation ='pos')
#You can change the values of the how many datapoints, variance, correlations and check the rsquared error to plot and visulaise how it behaves 
#with the changing values

#All the best!

