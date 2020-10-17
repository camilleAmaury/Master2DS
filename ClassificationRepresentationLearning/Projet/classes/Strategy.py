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
        distances = [possibilities[i][1] for i in range(len(possibilities))]
        ids = [possibilities[i][0] for i in range(len(possibilities))]
        ratios = self.metric.apply(distances, rewards)
        action = self.metric.choose(distances, ids, ratios, time_left)
        return int(ids[action]), distances[action]
        
    # string representation of the object
    def __repr__(self):
        return "<Object : Strategy (DataCamp) <> Naive_Metric_Strategy>"
    def __str__(self):
        return self.__repr__()


class OneStep_Metric_Strategy(Strategy):
    
    def __init__(self, metric):
        super(OneStep_Metric_Strategy, self).__init__()
        self.metric = metric
        
    def choose_action(self, possibilities, rewards, time_left, coordinate):
        distances = [possibilities[i][1] for i in range(len(possibilities))]
        ids = [possibilities[i][0] for i in range(len(possibilities))]
        ratios = self.metric.apply(distances, rewards)
        action = self.metric.choose(distances, ids, ratios, time_left)
        return int(ids[action]), distances[action]
        
    # string representation of the object
    def __repr__(self):
        return "<Object : Strategy (DataCamp) <> OneStep_Metric_Strategy>"
    def __str__(self):
        return self.__repr__()


