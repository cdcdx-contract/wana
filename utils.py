import six
import struct

# memory arguments
class memarg(object):
    def __init__(self, offset, align):
        self.offset = offset
        self.align = align

def is_real(value):
    return isinstance(value, six.integer_types)

def is_symbolic(value):
    return not isinstance(value, six.integer_types)

def read_u32(bytecode, pos):
    first, second, third, fourth = struct.unpack('BBBB', bytecode[pos:pos+5])
    if (first & 0x80) == 0:
        size = first & 0x7f
        byte_move = 1
    elif (second & 0x80) == 0: 
        size = ((first & 0x7f) << 7) | (first & 0x7f)
        byte_move = 2
    elif (third & 0x80) == 0:
        size = ((third & 0x7f) << 14) | ((second & 0x7f) << 7) | (first & 0x7f)
        byte_move = 3
    elif (fourth & 0x80) == 0:
        size = ((fourth & 0x7f) << 21) | ((third & 0x7f) << 14) | ((second & 0x7f) << 7) | (first & 0x7f)
        byte_move = 4
    elif (bytecode[pos+4] & 0x80) == 0:
        if bytecode[pos+4] & 0xf0:
            return (-1, -1)  # The size of bits > 32bits
        size = (bytecode[pos+4] & 0x7f) << 28) | ((fourth & 0x7f) << 21) | ((third & 0x7f) << 14) | ((second & 0x7f) << 7) | (first & 0x7f)
        byte_move = 5
    else:
        size = -1
        byte_move = -1
    
    return (size, byte_move)