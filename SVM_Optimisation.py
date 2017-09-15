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
        
        all_data = []
        for yi in self.data:
            for featureset  in self.data[yi]:
                for feature in featureset:
                    all_data.append(feature)
        
        self.max_feature_value = max(all_data)
        self.min_feature_value = min(all_data)
        all_data = None
        
        step_sizes = [self.max_feature_value * 0.1, 
                     self.max_feature_value * 0.01,
                      #point of expense:
                     self.max_feature_value * 0.001]
        
