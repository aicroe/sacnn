class InstanceStack(object):

    def __init__(self, max_stack):
        self._max_stack = max_stack
        self._stack = []

    def __contains__(self, name):
        return name in map(lambda instance: instance.get_arch_name(), self._stack)

    def __getitem__(self, name):
        for instance in self._stack:
            if instance.get_arch_name() == name:
                return instance
        return None

    def push(self, instance):
        self._stack.append(instance)
        if len(self._stack) >= self._max_stack:
            self._stack.pop(0)
