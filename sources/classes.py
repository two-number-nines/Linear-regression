import  matplotlib.pyplot as plt
from dataclasses import dataclass
import random


@dataclass
class LinearRegression:

    dependent:              list               # we try to predict (price)
    undependent:            list
    n_dependent:            float
    n_undependent:          float
    set_size:               int
    max_iteration:          int
    crit_convergence:       float
    learning_rate:          float
    t0 = random.uniform(0,1)                   # the weight
    t1 = random.uniform(0,1)                   # the bias

    def cost(self):
        return sum([(self.t0 + self.t1*self.undependent[i] - self.dependent[i])**2 for i in range(self.set_size)])

    def gradient_descent(self):
        grad0 = 1.0/self.set_size * sum([(self.t0 + self.t1*self.n_undependent[i] - self.n_dependent[i]) for i in range(self.set_size)]) 
        grad1 = 1.0/self.set_size * sum([(self.t0 + self.t1*self.n_undependent[i] - self.n_dependent[i])*self.n_undependent[i] for i in range(self.set_size)])

        return grad0, grad1

    def update_thetas(self, grad0, grad1):
        temp0 = self.t0 - self.learning_rate * grad0
        temp1 = self.t1 - self.learning_rate * grad1  
        self.t0 = temp0
        self.t1 = temp1

    def unnormalize_thetas(self):
        self.t1 = (max(self.dependent) - min(self.dependent)) * self.t1 / (max(self.undependent) - min(self.undependent))
        self.t0 = min(self.dependent) + self.t0 * (max(self.dependent) - min(self.dependent)) + self.t1 * (1 - min(self.undependent))


@dataclass
class PlotGraph:

    title:  str = 'Real Value'
    ylabel: str = 'Price'
    xlabel: str = 'Mileage'

    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)

    def __estimate_price(self, theta0, theta1, mileage):
        return (theta0 + (theta1 * mileage))

    def plot_basic_graph(self, x: list, y: list, t0: float, t1: float):
        plt.plot(x, y, 'ro')
        plt.plot([min(x), max(x)], [self.__estimate_price(t0, t1, min(x)), self.__estimate_price(t0, t1, max(x))])
        plt.show()
