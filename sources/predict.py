import click
from pathlib import Path
import logging
import csv

def estimate_price(theta0, theta1, mileage):
    return (theta0 + (theta1 * mileage))

@click.command()
@click.option('--mileage', prompt="Give the mileage for your car please", type=int)
def predict_price(mileage: int):
    p = Path(Path.cwd()/"thetas.csv")
    theta0 = 0
    theta1 = 0
    if p.exists():
        with open('thetas.csv', 'r') as f:
            read_data = csv.reader(f, delimiter=',')
            t0, t1 = next(read_data)
        result = estimate_price(float(t0), float(t1), mileage)
        print(result)
    else:
        train = input("You didn't train the model yet, want to train? y/n: \n")
        if train == 'y':
            print("we are going to train")
        else:
            estimatePrice = theta0 + (theta1 * mileage)
            print(estimatePrice)


if __name__ == "__main__":
    predict_price()
