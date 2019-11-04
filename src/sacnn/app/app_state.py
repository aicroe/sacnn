import sqlite3

from sacnn.core.fs_utils import get_app_dir


class _AppState(object):

    def __init__(self):
        self.cursor = sqlite3.connect(
            str(get_app_dir().joinpath('app_state')),
            check_same_thread=False,
        ).cursor()
        self.cursor.execute(
            'CREATE TABLE IF NOT EXISTS instances (name UNIQUE, hidden_units, num_labels, arch, PRIMARY KEY (name))'
        )

    def get_all_instances(self):
        return self.cursor.execute('SELECT * FROM instances')

    def is_unique_name(self, name):
        instances = self.cursor.execute(
            'SELECT name FROM instances WHERE name = ?',
            (name,),
        ).fetchall()
        return len(instances) == 0

    def record_instance(self, name, hidden_units, num_labels, arch):
        self.cursor.execute(
            'INSERT INTO instances VALUES (?, ?, ?, ?)',
            (name, hidden_units, num_labels, arch),
        )
        self.cursor.connection.commit()

    def remove_instance(self, name):
        self.cursor.execute('DELETE FROM instances WHERE name = ?', (name,))
        self.cursor.connection.commit()

    def get_instance_by_name(self, name):
        return self.cursor.execute(
            'SELECT * FROM instances WHERE name = ?',
            (name,),
        ).fetchone()

    def get_instance_names(self):
        return map(
            lambda row: row[0],
            self.cursor.execute('SELECT name FROM instances'),
        )

app_state = _AppState()
