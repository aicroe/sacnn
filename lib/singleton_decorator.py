def singleton(*args):
    def decorator(a_class):
        a_class.__instance = None

        def get_instance():
            if a_class.__instance is None:
                a_class.__instance = a_class(*args)
            return a_class.__instance

        a_class.get_instance = get_instance
        return a_class

    return decorator
