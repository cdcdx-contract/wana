import six


# memory arguments
class memarg(object):
    def __init__(self, offset, align):
        self.offset = offset
        self.align = align

def is_real(value):
    return isinstance(value, six.integer_types)

def is_symbolic(value):
    return not isinstance(value, six.integer_types)

def 