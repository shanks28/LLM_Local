# main.py
import argparse
from data import load_data
from train import train_rf
from predict import predict_random
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["train", "test"])
    parser.add_argument("--data", required=True)
    parser.add_argument("--artifacts", default="artifacts")
    args = parser.parse_args()

    os.makedirs(args.artifacts, exist_ok=True)

    X_train, X_test, y_train, y_test, preprocessor = load_data(args.data)

    if args.command == "train":
        train_rf(
            X_train, y_train, X_test, y_test, preprocessor, args.artifacts
        )

    elif args.command == "test":
        predict_random(X_test, y_test, args.artifacts)


if __name__ == "__main__":
    main()
