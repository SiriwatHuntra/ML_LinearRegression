import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from ipywidgets import interact

# loop 
#find dif between data val and estimate val
def cost_cal(x, y, w, b):
    """
    calculate cost
    x => data
    y => target
    w, b => parameter
    """

    n = x.shape[0]
    cost_sum = 0
    for i in range(n):
        f_wb = w * x[i] + b
        cost = (f_wb - y[i]) **2
        cost_sum += cost

    total_cost = (1/(2*n)) * cost_sum
    # slide 52, evaluation equation
    print(f"Cost is: {total_cost}")
    return total_cost

def plot(x, y, w, b):
    plt.scatter(x, y, marker='x', c='r')
    n = x.shape[0]
    f_wb = np.zeros(n)
    for i in range(n):
        f_wb[i] = w * x[i] + b

    plt.plot(x, f_wb, c='b', label='Predict')
    plt.ylabel('Y')
    plt.xlabel('X')
    plt.show()



def main():    
    #data
    x_train = np.array([0.0, 2.0])
    y_train = np.array([0.0, 2.0])

    #w, b parameter
    w = 1 #slope
    b = 0 #y axis cross

    #cost compute
    cost_cal(x_train, y_train, w, b)
    plot(x_train, y_train, w, b)

main()