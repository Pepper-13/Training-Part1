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


