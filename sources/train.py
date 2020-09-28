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

def update_thetas(theta_0, theta_1, mileage, price, total, learning_rate):
    theta_0_deriv = 0
    theta_1_deriv = 0

    for i in range(total):
        # Calculate partial derivatives
        # -2x(y - (mx + b))
        theta_0_deriv += -2*mileage[i] * (price[i] - (theta_0*mileage[i] + theta_1)) # weight

        # -2(y - (mx + b))
        theta_1_deriv += -2*(price[i] - (theta_0*mileage[i] + theta_1)) #  bias

    # We subtract because the derivatives point in direction of steepest ascent
    theta_0 -= (theta_0_deriv / total) * learning_rate
    theta_1 -= (theta_1_deriv / total) * learning_rate

    return theta_0, theta_1

def cost_function(theta_0, theta_1, mileage, price, total):
    '''this number represents how wrong the model is in terms 
    of its ability to estimate the relationship between X and y'''
    error = 0.0
    for i in range(total):
        error += (price[i] - (theta_0 * mileage[i] + theta_1))**2   #MSE 
    return error / total

def linear_regression(dataset: NDArray[(24, 2), Int[64]]):
    print("dataset:\n", dataset, "\n")
    learning_rate = 0.1
    theta_0 = 0.0                                            # the weight
    theta_1 = 0.0                                            # the bias
    mileage = [x[0] for x in dataset]
    price = [x[1] for x in dataset]
    total = len(dataset)
    cost_history = []
    # gradient descent
    for i in range(100):
        theta_0, theta_1 = update_thetas(theta_0, theta_1, mileage, price, total, learning_rate)

        #Calculate cost for auditing purposes
        cost = cost_function(theta_0, theta_1, mileage, price, total)
        cost_history.append(cost)

        # Log Progress
        if i % 10 == 0:
            print(f"iter={i}    theta_0={theta_0}    theta_1={theta_1}    cost={cost}")
    print("Final theta's: ", theta_0, theta_1)



def extract_data():
    data_file = Path(Path.cwd()/"ndata.csv")
    if data_file.exists():
        dataset = np.genfromtxt(data_file, delimiter=',', skip_header=1, dtype=float)
        linear_regression(dataset)
    else:
        logging.error(f"Could not find your dataset, should be stored as such: {Path.cwd()}/ndata.csv")


if __name__ == "__main__":
    extract_data()
