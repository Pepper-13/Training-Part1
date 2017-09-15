#SVM Optimisation algo from scratch

import matplotlib.pyplot as plt 
from matplotlib import style
import numpy as np
style.use('ggplot')

class Support_Vector_Machine:
    def _init_(self, visualization= True ):
        self.visualization = visualization
        self.colors = {1: 'r', -1 : 'b'}
        if self.visulization:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(1,1,1)
    #train
    def fit(self, data):
        self.data = data
        #{|w|: [w,b]}
        opt_dict = {}
        
        transforms = [[1,1],
                     [1,-1],
                     [-1,1],
                     [-1,-1]]
