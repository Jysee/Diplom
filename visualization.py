# visualization.py

import json
import csv
import matplotlib.pyplot as plt
import pandas as pd

def plot_metrics(metrics_csv: str, output_dir: str = 'figures'):
    df = pd.read_csv(metrics_csv, header=None,
                     names=['epoch', 'train_loss', 'val_loss'])

    df['epoch'] = pd.to_numeric(df['epoch'], errors='coerce')
    df['train_loss'] = pd.to_numeric(df['train_loss'], errors='coerce')
    df['val_loss'] = pd.to_numeric(df['val_loss'], errors='coerce')

    plt.figure(figsize=(8,6))
    plt.plot(df['epoch'], df['train_loss'], label='Train Loss')
    plt.plot(df['epoch'], df['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig(f'{output_dir}/loss_curve.png')
    plt.close()


if __name__ == "__main__":
    import argparse, os
    parser = argparse.ArgumentParser()
    parser.add_argument('--metrics_csv', type=str, default='results/metrics.csv')
    parser.add_argument('--output_dir', type=str, default='figures')
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    plot_metrics(args.metrics_csv, args.output_dir)