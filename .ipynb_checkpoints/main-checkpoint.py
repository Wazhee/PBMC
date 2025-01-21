import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-train', action='store_true')
parser.add_argument('-model', default='unet', choices=['unet', 'attention', 'residual'])
parser.add_argument('-dataset', default='fib', choices=['fib'])
parser.add_argument('-augment', action='store_true') # enable data augmentation
parser.add_argument('-early_stopping', action='store_true') # enable early_stopping

args = parser.parse_args()
model = args.model
dataset = args.dataset
early_stopping = args.early_stopping


if __name__ == "__main__":
    print(model, dataset, early_stopping)