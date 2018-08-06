
class InstanceStack(object):
    def __init__(self, max_stack):
        self.max_stack = max_stack
        self.stack = []

    def __contains__(self, name):
        return name in map(lambda instance: instance.name, self.stack)

    def __getitem__(self, name):
        for instance in self.stack:
            if instance.name == name:
                return instance
        return None

    def push(self, instance):
        self.stack.append(instance)
        if len(self.stack) >= self.max_stack:
            self.stack.pop(0)
