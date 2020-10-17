import numpy as np
import math

# ----------------------------------- BASE ----------------------------------- #

class Metric(object):
    def __init__(self):
        super(Metric, self).__init__()
        
    def apply(self, distances, rewards):
        return None
    
    def choose(self, distances, ids, ratios, time_left):
        target_pos = np.argmax(ratios)
        if time_left - distances[target_pos] < 0:
            # the metrics can't find a cell to eat in the given left time
            return -1
        else:
            return target_pos
    
    # string representation of the object
    def __repr__(self):
        return "<Object : Metric (DataCamp)>"
    def __str__(self):
        return "Metric"

# ----------------------------------- CHILDREN ----------------------------------- #

class Euclidean_1_Metric(Metric):
    def __init__(self):
        super(Euclidean_1_Metric, self).__init__()
        
    def apply(self, distances, rewards):
        n = len(distances)
        euclidean_ratios = np.array([ (rewards[i] / distances[i]**3 ) if distances[i] != 0. and rewards[i] != 0. else 0. for i in range(n)])
        mini = np.min(euclidean_ratios)
        if mini < 0.:
            for i in range(n):
                if euclidean_ratios[i] != 0.:
                    euclidean_ratios[i] -= (mini - 1)
        return euclidean_ratios
    
class Euclidean_2_Metric(Metric):
    def __init__(self):
        super(Euclidean_2_Metric, self).__init__()
        
    def apply(self, distances, rewards):
        n = len(distances)
        euclidean_ratios = np.array([( (math.log(rewards[i]) - math.log(distances[i])) / distances[i] ) if distances[i] != 0. and rewards[i] != 0. else 0. for i in range(n)])
        mini = np.min(euclidean_ratios)
        if mini < 0.:
            for i in range(n):
                if euclidean_ratios[i] != 0.:
                    euclidean_ratios[i] -= (mini - 1)
        return euclidean_ratios
    
class Euclidean_3_Metric(Metric):
    def __init__(self):
        super(Euclidean_3_Metric, self).__init__()
        
    def apply(self, distances, rewards):
        n = len(distances)
        euclidean_ratios = np.array([( (math.sqrt(rewards[i]) - math.sqrt(distances[i])) / distances[i] ) if distances[i] != 0. and rewards[i] != 0. else 0. for i in range(n)])
        mini = np.min(euclidean_ratios)
        if mini < 0.:
            for i in range(n):
                if euclidean_ratios[i] != 0.:
                    euclidean_ratios[i] -= (mini - 1)
        return euclidean_ratios


class OneStep_Euclidean_1_Metric(Metric):
    def __init__(self, ratio):
        super(OneStep_Euclidean_1_Metric, self).__init__()
        self.ratio = ratio
        
    def apply(self, distances, rewards):
        n = len(distances)
        euclidean_ratios = np.array([ ((rewards[i][0] / distances[i]**3 )*(self.ratio) + rewards[i][1]*(1-self.ratio))
                                       if distances[i] != 0. and rewards[i][0] != 0. else 0. for i in range(n)])
        mini = np.min(euclidean_ratios)
        if mini < 0.:
            for i in range(n):
                if euclidean_ratios[i] != 0.:
                    euclidean_ratios[i] -= (mini - 1)
        return euclidean_ratios