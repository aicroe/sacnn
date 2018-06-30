import numpy as np
from pathlib import Path


class DataSaver(object):

    @staticmethod
    def create_dir(dir_path):
        return Path(dir_path).mkdir(parents=True, exist_ok=True)

    @staticmethod
    def get_app_dir():
        user_dir = Path.home()
        app_dir = user_dir.joinpath('.sacnn')
        DataSaver.create_dir(app_dir)
        return app_dir

    @staticmethod
    def prepare_dir(dir_name):
        app_dir = DataSaver.get_app_dir()
        full_dir = app_dir.joinpath(dir_name)
        DataSaver.create_dir(full_dir)
        return full_dir

    @staticmethod
    def _save_data(dir_name,
                   train_dataset,
                   train_labels,
                   val_dataset,
                   val_labels,
                   test_dataset,
                   test_labels):
        save_dir = DataSaver.prepare_dir(dir_name)
        np.save(str(save_dir.joinpath('train_dataset.npy')), train_dataset)
        np.save(str(save_dir.joinpath('train_labels.npy')), train_labels)
        np.save(str(save_dir.joinpath('val_dataset.npy')), val_dataset)
        np.save(str(save_dir.joinpath('val_labels.npy')), val_labels)
        np.save(str(save_dir.joinpath('test_dataset.npy')), test_dataset)
        np.save(str(save_dir.joinpath('test_labels.npy')), test_labels)

    @staticmethod
    def save_data(train_dataset,
                  train_labels,
                  val_dataset,
                  val_labels,
                  test_dataset,
                  test_labels):
        DataSaver._save_data('data', 
                             train_dataset,
                             train_labels,
                             val_dataset,
                             val_labels,
                             test_dataset,
                             test_labels)

    @staticmethod
    def _load_data(dir_name, just_test_data=False):
        save_dir = DataSaver.prepare_dir(dir_name)
        test_dataset = np.load(str(save_dir.joinpath('test_dataset.npy')))
        test_labels = np.load(str(save_dir.joinpath('test_labels.npy')))
        if just_test_data:
            return test_dataset, test_labels
        
        train_dataset = np.load(str(save_dir.joinpath('train_dataset.npy')))
        train_labels = np.load(str(save_dir.joinpath('train_labels.npy')))
        val_dataset = np.load(str(save_dir.joinpath('val_dataset.npy')))
        val_labels = np.load(str(save_dir.joinpath('val_labels.npy')))
        return train_dataset, train_labels, val_dataset, val_labels, test_dataset, test_labels

    @staticmethod
    def load_data(just_test_data=False):
        return DataSaver._load_data('data', just_test_data)

    @staticmethod
    def save_reduced_data(train_dataset,
                          train_labels,
                          val_dataset,
                          val_labels,
                          test_dataset,
                          test_labels):
        DataSaver._save_data('data_reduced', 
                             train_dataset,
                             train_labels,
                             val_dataset,
                             val_labels,
                             test_dataset,
                             test_labels)

    @staticmethod
    def load_reduced_data(just_test_data=False):
        return DataSaver._load_data('data_reduced', just_test_data)
