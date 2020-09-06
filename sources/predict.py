import click
# estimatePrice(mileage) = θ0 + (θ1 ∗ mileage)
# what is theta

@click.command()
@click.prompt()
def predict_price():
    print("Hello Predict")

if __name__ == "__main__":
    predict_price()