import click
from pathlib import Path
import logging

def estimate_price(theta0, theta1, mileage):
    return (theta0 + (theta1 * mileage))

@click.command()
@click.option('--mileage', prompt="Give the mileage for your car please", type=int)
def predict_price(mileage: int):
    print(f"Hello mileage = {mileage}")
    p = Path(Path.cwd()/"thetas.csv")
    theta0 = 0
    theta1 = 0
    if p.exists():
        print("ready to give results here because thetas.csv exists")
    else:
        train = input("You didn't train the model yet, want to train? y/n: \n")
        if train == 'y':
            print("we are going to train")
        else:
            estimatePrice = theta0 + (theta1 * mileage)
            print(estimatePrice)


if __name__ == "__main__":
    predict_price()
