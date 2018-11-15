import six


def is_real(value):
    return isinstance(value, six.integer_types)

def is_symbolic(value):
    return not isinstance(value, six.integer_types)

def 