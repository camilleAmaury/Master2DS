import numpy as np
import math

# ----------------------------------- BASE ----------------------------------- #
class Strategy(object):
    
    def __init__(self):
        super(Strategy, self).__init__()
    
    def choose_action(self, possibilities, rewards, time_left, coordinate):
        return None
        
    # string representation of the object
    def __repr__(self):
        return "<Object : Strategy (DataCamp)>"
    def __str__(self):
        return self.__repr__()

# ----------------------------------- CHILDREN ----------------------------------- #
class Naive_Metric_Strategy(Strategy):
    
    def __init__(self, metric):
        super(Naive_Metric_Strategy, self).__init__()
        self.metric = metric
        
    def choose_action(self, possibilities, rewards, time_left, coordinate):
        distances = np.zeros((possibilities.shape[0],))
        for point in range(possibilities.shape[0]):
            distances[point] = math.sqrt((coordinate[0]- possibilities[point,0])**2 + (coordinate[1] - possibilities[point,1])**2)
        ratios = self.metric.apply(distances, rewards)
        action = self.metric.choose(distances, ratios, time_left)
        return action, distances[action]
        
    # string representation of the object
    def __repr__(self):
        return "<Object : Strategy (DataCamp) <> Naive_Metric_Strategy>"
    def __str__(self):
        return self.__repr__()


