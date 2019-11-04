import numpy as np
from pathlib import Path

APP_NAME = 'sacnn'

def create_dir(dir_path):
    return Path(dir_path).mkdir(parents=True, exist_ok=True)

def get_app_dir():
    user_dir = Path.home()
    return user_dir.joinpath('.%s' % APP_NAME)

def prepare_dir(dir_name):
    app_dir = get_app_dir()
    full_path = app_dir.joinpath(dir_name)
    create_dir(full_path)
    return full_path

def save_np_tensor(name, tensor, dir_name):
    save_dir = prepare_dir(dir_name)
    save_path = str(save_dir.joinpath('%s.npy' % name))
    np.save(save_path, tensor)

def load_np_tensor(name, dir_name):
    save_dir = prepare_dir(dir_name)
    save_path = str(save_dir.joinpath('%s.npy' % name))
    return np.load(save_path)
