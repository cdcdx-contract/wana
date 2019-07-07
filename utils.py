#!/usr/bin/python3

import six
import struct
import z3

def is_int(value: int) -> bool:
    return isinstance(value, six.integer_types)

def is_float(value: float) -> bool:
    return isinstance(value, float)

def is_symbolic(value: 'Value') -> bool:
    return not isinstance(value, six.integer_types) and not isinstance(value, float)

def is_all_real(*args) -> bool:
    for elem in args:
        if is_symbolic(elem):
            return False
    return True

def to_symbolic(number: int, len: int) -> 'BitVecVal':
    if is_int(number):
        return z3.BitVecVal(number, len)
    return number

def to_signed(number: int, len: int) -> int:
    if number > 2**(len - 1):
        return (2**len - number) * (-1)
    else:
        return number

def to_unsigned(number: int, len: int) -> int:
    if number < 0:
        return number + 2**len
    else:
        return number

def check_sat(solver: 'Solver', pop_if_exception: bool=True) -> 'State':
    try:
        ret = solver.check()
        if ret == z3.unknown:
            raise z3.Z3Exception(solver.reason_unknown())
    except Exception as e:
        if pop_if_exception:
            solver.pop()
        raise e
    return ret