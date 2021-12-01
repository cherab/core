# Copyright (c) 2015, Dr Alex Meakins
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     1. Redistributions of source code must retain the above copyright notice,
#        this list of conditions and the following disclaimer.
#
#     2. Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in the
#        documentation and/or other materials provided with the distribution.
#
#     3. The name of the author may not be used to endorse or promote products
#        derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

from weakref import ref
from types import MethodType, BuiltinMethodType


class Notifier:
    """
    Allows objects to broadcast notifications to observing objects.

    This object implements a version of the observer pattern. Objects wishing
    to be notified may register a callback function with the notifier. The
    callbacks will be called when the notify method of the Notifier is called.

    The primary purpose of this class is to permit cache control between
    disconnected objects. To speed up calculations, objects may cache the
    results of a calculation involving a source object. If the source object
    data changes, the caches of the dependent objects must be invalidated
    otherwise stale data may be used in subsequent calculations.

    Callbacks are assumed to have no arguments.

    This object holds weak references to callbacks. If an observing object has
    registered a method as a callback and that object is subsequently deleted,
    the callback will be automatically removed from the list of registered
    callbacks. The Notifier will not prevent referenced objects being garbage
    collected.
    """

    def __init__(self):

        self._callbacks_refs = []

    def add(self, callback):

        if self.is_present(callback):
            return

        # instance methods been special handling
        if isinstance(callback, (MethodType, BuiltinMethodType)):
            self._callbacks_refs.append((ref(callback.__self__), callback.__name__))
        else:
            self._callbacks_refs.append(ref(callback))

    def remove(self, callback):

        if isinstance(callback, (MethodType, BuiltinMethodType)):
            self._remove_method(callback)
        else:
            self._remove_callable(callback)

    def is_present(self, callback):

        if isinstance(callback, (MethodType, BuiltinMethodType)):
            return self._is_present_method(callback)
        else:
            return self._is_present_callable(callback)

    def notify(self):

        dead_callbacks = []

        # trigger callbacks for each observer
        for reference in self._callbacks_refs:

            # obtain callback from weak reference
            if isinstance(reference, tuple):

                instance = reference[0]()

                # does the object still exist
                if instance is None:
                    dead_callbacks.append(reference)
                    continue

                # bound method
                method = instance.__getattribute__(reference[1])

                # call method
                method()

            else:

                callback = reference()

                # does the callback object still exist
                if callback is None:
                    dead_callbacks.append(reference)
                    continue

                callback()

        self._purge(dead_callbacks)

    def _remove_method(self, callback):

        for reference in self._callbacks_refs:
            if isinstance(reference, tuple):
                instance = reference[0]()
                method = reference[1]                
                if callback.__self__ == instance and callback.__name__ == method:
                    self._purge([reference])
                    break

    def _remove_callable(self, callback):

        for reference in self._callbacks_refs:
            if not isinstance(reference, tuple):
                if reference() == callback:
                    self._purge([reference])
                    break

    def _purge(self, refs):

        for ref in refs:
            self._callbacks_refs.remove(ref)

    def _is_present_method(self, callback):

        for reference in self._callbacks_refs:
            if isinstance(reference, tuple):
                instance = reference[0]()
                method = reference[1]
                if callback.__self__ == instance and callback.__name__ == method:
                    return True

        return False

    def _is_present_callable(self, callable):

        for reference in self._callbacks_refs:
            if not isinstance(reference, tuple):
                if callable is reference():
                    return True

        return False


class NotifyingList(list):
    """
    A list that reports changes to its contents.

    The NotifyingList class is a subclass of the builtin list type. It extends
    the list type to add a Notifier object that generates notifications
    whenever the list contents are modified. A notifier attribute is provided
    to supply access to configure the internal Notifier object.

    The NotifierList implements the entire list interface. Please note however
    that __add__ or __mul__ operations involving a NotifyingList will return
    a basic builtin list.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._notifier = Notifier()

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        self._notifier.notify()

    def __delitem__(self, key):
        super().__delitem__(key)
        self._notifier.notify()

    def __iadd__(self, other):
        super().__iadd__(other)
        self._notifier.notify()
        return self

    def __imul__(self, other):
        super().__imul__(other)
        self._notifier.notify()
        return self

    @property
    def notifier(self):
        return self._notifier

    def append(self, p_object):
        super().append(p_object)
        self._notifier.notify()

    def insert(self, index, p_object):
        super().insert(index, p_object)
        self._notifier.notify()

    def reverse(self):
        super().reverse()
        self._notifier.notify()

    def extend(self, iterable):
        super().extend(iterable)
        self._notifier.notify()

    def pop(self, index=-1):
        item = super().pop(index)
        self._notifier.notify()
        return item

    def remove(self, value):
        super().remove(value)
        self._notifier.notify()

    def sort(self, **kwargs):
        super().sort(**kwargs)
        self._notifier.notify()

    def clear(self):
        super().clear()
        self._notifier.notify()


