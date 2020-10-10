import click
from pathlib import Path
import logging
import csv
from sources.train import initialize_model

def estimate_price(theta0, theta1, mileage):
    return (theta0 + (theta1 * mileage))

def read_and_print(mileage):
    with open('thetas.csv', 'r') as f:
        read_data = csv.reader(f, delimiter=',')
        t0, t1 = next(read_data)
    print(f"The estimated price is: ${estimate_price(float(t0), float(t1), mileage)}")


@click.command()
@click.option('--mileage', prompt="Give the mileage for your car please", type=int)
def predict_price(mileage: int):
    p = Path(Path.cwd()/"thetas.csv")
    t0 = 0
    t1 = 0
    if p.exists():
        read_and_print(mileage)
    else:
        prompt = input("The model is not trained yet, want to train? y/n\n")
        if prompt == 'y':
            initialize_model()
            read_and_print(mileage)
        else:
            print(f"The estimated price is: ${estimate_price(t0, t1, mileage)}")


if __name__ == "__main__":
    predict_price()
