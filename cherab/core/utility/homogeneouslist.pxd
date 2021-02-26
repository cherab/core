cdef class HomogeneousList:

    cdef:
        readonly object item_class
        readonly int length
        readonly object notifier
        list _list
    
    cpdef list get_list(self)
    
    cpdef int get_length(self)