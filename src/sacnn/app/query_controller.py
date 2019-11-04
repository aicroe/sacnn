from .app_state import app_state

class QueryController():

    def get_instances_names(self):
        return app_state.get_instance_names()

    def is_unique_instance_name(self, name):
        return app_state.is_unique_name(name)
