
import inspect

from cherab.core.utility import Notifier

cdef class HomogeneousList:

    def __init__(self, item_class):

        if not inspect.isclass(item_class):
            raise TypeError("item_class has to be a python class")

        self.item_class = item_class
        self._list = []
        self.length = 0
        self.notifier = Notifier()

    def __getitem__(self, item):
        return self._list[item]

    def __iter__(self):
        return iter(self._list)

    def set(self, new_list):

        # copy models and test it is an iterable
        new_list = list(new_list)

        # check contents of list are beam models
        for item in new_list:
            if not isinstance(item, self.item_class):
                raise TypeError('The list must consist of only {} instances.'.format(self.item_class))
        
        self._list = new_list
        self.length = len(self._list)
        self.notifier.notify()
    
    def add(self, item):
    
        if not isinstance(item, self.item_class):
                raise TypeError('The item must be instance of {}.'.format(self.item_class))

        self._list.append(item)
        self.length = len(self._list)
        self.notifier.notify()

    def clear(self):
        self._list = []
        self.length = 0
        self.notifier.notify()
    
    def remove(self, item):
        self._list.remove(item)
        self.length = len(self._list)
        self.notifier.notify()

    cpdef list get_list(self):
        return self._list
    
    cpdef int get_length(self):
        return self.length


