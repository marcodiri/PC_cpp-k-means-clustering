
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
        with open(filepath, 'w') as f:
            np.savetxt(f, data)
        print("created "+filepath)


def load_data(filename: str, dtype=float):
    filepath = DATA_PATH+filename
    if os.path.isfile(filepath):
        with open(filepath, 'r') as f:
            return np.loadtxt(f, dtype=dtype)
