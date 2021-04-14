
import os
import numpy as np

ROOT_DIR = os.path.dirname(os.getcwd()) + '/'
DATA_PATH = ROOT_DIR + 'data/'


def touch_dir(base_dir: str) -> None:
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)


def save_data(data: np.ndarray, filename: str):
    touch_dir(DATA_PATH)
    filepath = DATA_PATH+filename
    if os.path.isfile(filepath):
        print(filepath+" already exists")
    else:
        if data.dtype == np.float64:
            with open(filepath, 'w') as f:
                np.savetxt(f, data.astype(float), fmt='%f')
                print("created "+filepath)
        elif data.dtype == np.int64:
            with open(filepath, 'w') as f:
                np.savetxt(f, data.astype(int), fmt='%i')
                print("created "+filepath)
        else:
            print("format not supported")


def load_data(filename: str, dtype=float):
    filepath = DATA_PATH+filename
    if os.path.isfile(filepath):
        with open(filepath, 'r') as f:
            return np.loadtxt(f, dtype=dtype)
