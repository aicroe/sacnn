from lib.data_saver import DataSaver
import sqlite3


class AppState(object):
    def __init__(self):
        self.cursor = sqlite3.connect(
            str(DataSaver.get_app_dir().joinpath('app_state')), check_same_thread=False).cursor()
        self.cursor.execute(
            'CREATE TABLE IF NOT EXISTS instances (name UNIQUE, hidden_units, num_labels, arch, PRIMARY KEY (name))')

    def get_all_instances(self):
        return self.cursor.execute('SELECT * FROM instances')

    def record_instance(self, name, hidden_units, num_labels, arch):
        self.cursor.execute(
            'INSERT INTO instances VALUES (?, ?, ?, ?)',
            (name, hidden_units, num_labels, arch))
        self.cursor.connection.commit()

    def get_instance_by_name(self, name):
        return self.cursor.execute(
            'SELECT * FROM instances WHERE name = ?', (name,)).fetchone()

    def get_instance_names(self):
        return map(lambda row: row[0],
                   self.cursor.execute('SELECT name FROM instances'))
