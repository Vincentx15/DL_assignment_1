import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse


def generate_learning_curves(log_path, output_path, metric, loss=True):
    """
    Read from logs json file and output learning curves at specified path.

    """

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    plt.figure(figsize=(8, 5))

    try:
        with open(log_path) as f:
            df = pd.read_json(f, lines=True)
    except FileNotFoundError as e:
        print("The log path is incorrect!")
        raise e

    if metric == 'accuracy':
        results = list(df['accuracy'].where(
            df['phase'] == 'valid').dropna())
        plt.ylabel(r'Accuracy (\%)')
    else:
        results = list(df['loss'].where(df['phase'] == 'valid').dropna())
        plt.ylabel(r'Loss (\%)')

    plt.xlabel('Epochs')

    plt.plot(results)
    plt.savefig(output_path + metric)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument("metric")
    args = parser.parse_args()
    generate_learning_curves(args.input, args.output, args.metric)
