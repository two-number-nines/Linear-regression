import numpy as np
from pathlib import Path
import logging
from nptyping import NDArray, Int

logging.basicConfig(level=logging.INFO)

# gradient descent is for updating the theta's
# learningRate which you can choose yourself (it a hyperparameter aka you can change it manually)
# you can change it because if you make it bigger then that average difference you calculated
# and will make bigger leaps to adjust
# convergence

def linear_regression(dataset: NDArray[(24, 2), Int[64]]):
    print(dataset)
    converged = False
    iter = 0
    ep = 1
    max_iter = 100
    learning_rate = 0.1
    t0 = 0.01                                            # the weight
    t1 = 0.01                                            # the bias
    mileage = [x[0] for x in dataset]
    price = [x[1] for x in dataset]
    total = len(dataset)
    cost = sum([(t0 + t1*mileage[i] - price[i])**2 for i in range(total)])
    # gradient descent
    while not converged:
        # for each training sample, compute the gradient (d/d_theta j(theta))
        grad0 = 1.0/total * sum([(t0 + t1*mileage[i] - price[i]) for i in range(total)]) 
        grad1 = 1.0/total * sum([(t0 + t1*mileage[i] - price[i])*mileage[i] for i in range(total)])
        print(grad0)
        # print(grad1)
        # exit()

        # update the theta_temp
        temp0 = t0 + learning_rate * grad0
        temp1 = t1 + learning_rate * grad1
    
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
    print("Final theta's: ", t0, t1)



def extract_data():
    data_file = Path(Path.cwd()/"data.csv")
    if data_file.exists():
        dataset = np.genfromtxt(data_file, delimiter=',', skip_header=1, dtype=int)
        linear_regression(dataset)
    else:
        logging.error(f"Could not find your dataset, should be stored as such: {Path.cwd()}/ndata.csv")


if __name__ == "__main__":
    extract_data()
