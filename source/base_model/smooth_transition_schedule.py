import numpy as np

def smooth_transition_schedule(start, end, steepness=10.0):
    midpoint = (start + end) / 2
    width = end - start

    def schedule(epoch):
        x = (epoch - midpoint) / (width / 2)
        return float(1 / (1 + np.exp(-steepness * x)))  # logistic sigmoid

    return schedule