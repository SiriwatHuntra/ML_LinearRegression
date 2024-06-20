#import lib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gsp

#coeficients estimate 
def estimate_coef(x, y):
    #num of observes point
    n = np.size(x)

    #x, y mean vector
    m_x = np.mean(x)
    m_y = np.mean(y)

    #cal cross-deviation, x deviation
    SS_xy = np.sum(x*y) - n*m_y*m_x
    SS_xx = np.sum(x*x) - n*m_x*m_x

    #cal coef
    b_1 = SS_xy/SS_xx
    b_0 = m_y - b_1*m_x
    return (b_0, b_1) 

def plot_reg_line(x, y, b):
    #plot point
    plt.scatter(x, y, color = 'm', s=30, marker = "o")

    #predict response vetor
    y_pred = b[0] + b[1]*x
    plt.plot(x, y_pred, color = 'g')

    #putting labels
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

def main():
    #data
    x = np.array([1,2,3,4,5])
    y = np.array([1,2,3,4,5])

    #etimate coef
    b = estimate_coef(x, y)
    #truple unpack for print
    b_0, b_1 = b

    plot_reg_line(x, y, b)
    print("coef: b_0 = {} b_1 = {}".format(b_0, b_1))

    

main()