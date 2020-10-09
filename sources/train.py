import numpy as np
from pathlib import Path
import logging
from nptyping import NDArray, Int
import random
import  matplotlib.pyplot as plt
from sources.predict import estimate_price

logging.basicConfig(level=logging.INFO)


def linear_regression(dataset: NDArray[(24, 2), Int[64]]):
    print(dataset)
    converged = False
    iter = 0
    ep = 0.000000000000001
    max_iter = 10000
    learning_rate = 0.5
    t0 = random.uniform(0,1)                                            # the weight
    t1 = random.uniform(0,1)                                            # the bias
    mileage1 = [x[0] for x in dataset]
    mileage = [float(i)/max(mileage1) for i in mileage1]
    price1 = [x[1] for x in dataset]
    price = [float(i)/max(price1) for i in price1]
    # other way of doing it
    # mileage = [(float(i) - min(mileage1)) / (max(mileage1) - min(mileage1)) for i in mileage1]
    # price = [(float(i) - min(price1)) / (max(price1) - min(price1)) for i in price1]
    total = len(dataset)
    cost = sum([(t0 + t1*mileage[i] - price[i])**2 for i in range(total)])
    # gradient descent
    while not converged:
        # for each training sample, compute the gradient (d/d_theta j(theta))
        grad0 = 1.0/total * sum([(t0 + t1*mileage[i] - price[i]) for i in range(total)]) 
        grad1 = 1.0/total * sum([(t0 + t1*mileage[i] - price[i])*mileage[i] for i in range(total)])

        # update the theta_temp
        temp0 = t0 - learning_rate * grad0
        temp1 = t1 - learning_rate * grad1
    
        # update theta
        t0 = temp0
        t1 = temp1

        # mean squared error
        new_cost = sum([(t0 + t1*mileage[i] - price[i])**2 for i in range(total)])

        if abs(cost - new_cost) <= ep:
            print ('Converged, iterations: ', iter, '!!!')
            converged = True
    
        cost = new_cost   # update error 
        iter += 1         # update iter
    
        if iter == max_iter:
            print ('Max interactions exceeded!')
            converged = True

        print(f"iter={iter} theta_0={t0} theta_1={t1} cost={cost}")

    print("\n\nFinal theta's (with standardization): ", t0, t1)

    # t0 = t0*max(mileage1)
    # t1 = t1*max(price1)
    t1 = (max(price1) - min(price1)) * t1 / (max(mileage1) - min(mileage1))
    t0 = min(price) + t0 * (max(price1) - min(price1)) + t1 * (1 - min(mileage1))
    print("Final theta's:                        ", t0, t1)


    plt.title('Real values')
    plt.ylabel('Price')
    plt.xlabel('Mileage')
    plt.plot(mileage1, price1, 'ro')
    plt.plot([min(mileage1), max(mileage1)], [estimate_price(t0, t1, min(mileage1)), \
            estimate_price(t0, t1, max(mileage1))])
    plt.show()




def extract_data():
    data_file = Path(Path.cwd()/"data.csv")
    if data_file.exists():
        dataset = np.genfromtxt(data_file, delimiter=',', skip_header=1, dtype=int)
        linear_regression(dataset)
    else:
        logging.error(f"Could not find your dataset, should be stored as such: {Path.cwd()}/ndata.csv")


if __name__ == "__main__":
    extract_data()
