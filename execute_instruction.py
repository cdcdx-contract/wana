from instructions import *
from utils import *
from math import *

# Executing the singal instruction
def execute_one_instruction(stack, instruction, locals_value, globals_value):
    op = opcode[instruction[0]]
    top = len(stack)

    if op == 'unreachable':
        pass

    elif op == 'nop':
        pass

    elif op == 'loop':
        pass

    elif op == 'if':
        pass
    
    elif op == 'else':
        pass

    elif op == 'end':
        pass

    elif op == 'br':
        pass

    elif op == 'br_if':
        pass

    elif op == 'br_table':
        pass

    elif op == 'return':
        pass

    elif op == 'call':
        pass

    elif op == 'call_indirect':
        pass

    elif op == 'drop':
        stack.pop()
        top -= 1

    elif op == 'select':
        # TODO
        # How to fork two different paths
        flag = stack.pop()
        val_2 = stack.pop()
        val_1 = stack.pop()
        if flag:
            stack.append(val_1)
        else:
            stack.append(val_2)
        top -= 2


    elif op == 'get_local':
        locals_pos = instruction[1]
        stack.append(locals_value[local_pos])
        top += 1

    elif op == 'set_local':
        locals_pos = instruction[1]
        if locals_value[local_pos] == stack[top]:
            stack.pop()
            top -= 1
        else:
            pass

    elif op == 'tee_local':
        locals_pos = instruction[1]
        value = stack.pop()
        stack.append(value)
        stack.append(value)
        top += 1
        if locals_value[local_pos] == stack[top]:
            stack.pop()
            top -= 1
        else:
            pass

    elif op == 'get_global':
        globals_pos = instruction[1]
        stack.append(globals_value[globals_pos])
        top += 1

    elif op == 'set_global':
        globals_pos = instruction[1]
        if globals_value[globals_pos] = stack.pop():
            stack.pop()
            top -= 1
        else:
            pass

    elif op == 'i32.load':
        align = instruction[1]
        offset = instruction[2]

        if mems[0] and 2**align <= 4:
            pass

    elif op == 'i64.load':
        align = instruction[1]
        offset = instruction[2]
        if mems[0] and 2**align <= 8:
            pass

    elif op == 'f32.load':
        pass

    elif op == 'f64.load':
        pass

    elif op == 'i32.load8_s':
        pass

    elif op == 'i32.load8_u':
        pass

    elif op == 'i32.load16_s':
        pass

    elif op == 'i32.load16_u':
        pass

    elif op == 'i64.load8_s':
        pass

    elif op == 'i64.load8_u':
        pass

    elif op == 'i64.load16_s':
        pass

    elif op == 'i64.load16_u':
        pass
    
    elif op == 'i32.store':
        pass

    elif op == 'i64.store':
        pass

    elif op == 'f32.store':
        pass

    elif op == 'f64.store':
        pass

    elif op == 'f64.store':
        pass

    elif op == 'i32.store8':
        pass

    elif op == 'i32.store16':
        pass

    elif op == 'i64.store8':
        pass

    elif op == 'i64.store16':
        pass

    elif op == 'i64.store32':
        pass

    elif op == 'memory.size':
        pass

    elif op == 'memory.grow':
        pass

    elif op == 'i32.const':
        i32_value = int(instruction[1], base=16)
        stack.append(i32_value)
        top += 1

    elif op == 'i64.const':
        i64_value = int(instruction[1], base=16)
        stack.append(i64_value)
        top += 1

    elif op == 'f32.const':
        pass

    elif op == 'f64.const':
        pass

    elif op in ['i32.eqz', 'i64.eqz']:
        stack[top-1] = 1 if stack[top-1] == 0 else 1

    elif op in ['i32.eq', 'i64.eq', \
                'f32.eq', 'f64.eq']:
        stack[top-2] = 1 if stack[top-2] == stack[top-1] else 0
        top -= 1

    elif op in ['i32.ne', 'i64.ne', \
                'f32.ne', 'f64.ne']:
        stack[top-2] = 1 if stack[top-2] != stack[top-1] else 0
        top -= 1

    elif op in ['i32.lt_s', 'i32.lt_u', 'i64.lt_s', 'i64.lt_u', \
                'f32.lt', 'f64.lt']:
        stack[top-2
        ] = 1 if stack[top-2] < stack[top-1] else 0
        top -= 1

    elif op in ['i32.gt_s', 'i32.gt_s', 'i64.gt_s', 'i64.gt_u', \
                'f32.gt', 'f64.gt']:
        stack[top-2] = 1 if stack[top-2] > stack[top-1] else 0
        top -= 1

    elif op in ['i32.le_s', 'i32.le_u', 'i64.le_s', 'i64.le_u', \
                'f32.le', 'f64.le']:
        stack[top-2] = 1 if stack[top-2] <= stack[top-1] else 0
        top -= 1

    elif op in ['i32.ge_s', 'i32.ge_u', 'i64.ge_s', 'i64.ge_u', \
                'f32.ge', 'f64.ge']:
        stack[top-2] = 1 if stack[top-2] >= stack[top-1] else 0
        top -= 1
    
    elif op in ['i32.clz', 'i64.clz']:
        if stack[top-1] == 0:
            stack[top-1] = int(op[1:3])
        else:
            stack[top-1] = int(op[1:3]) - bin(stack[top-1]) + 2  # Binary format '0b***'
    
    elif op in ['i32.ctz', 'i64.ctz']:
        if stack[top-1] == 0:
            stack[top-1] = int(op[1:3])
        else:
            binary_num = bin(stack[top-1])
            stack[top-1] = 0
            for i in range(len(binary_num)-1, -1, -1):
                if binary_num[i] != '0':
                    break
                stack[top-1] += 1

    
    elif op in ['i32.popcnt', 'i64.popcnt']:
        binary_num = bin(stack[top-1])[2:]  # Binary format '0b' is useless
        stack[top-1] = 0
        for elem in binary_num:
            if elem != '0':
                stack[top-1] += 1
    
    elif op in ['i32.add', 'i64.add']:
        stack[top-2] = stack[top-1] + stack[top-2]
        top -= 1

    elif op in ['i32.sub', 'i64.sub']:
        modulo = 2**int(op[1:3])
        stack[top-2] = (stack[top-2] - stack[top-1] + modulo) % modulo
        top -= 1

    elif op in ['i32.mul', 'i64.mul']:
        modulo = 2**int(op[1:3])
        stack[top-2] = (stack[top-2] * stack[top-1]) % modulo
        top -= 1

    elif op in ['i32.div_s', 'i32.div_u', 'i64.div_s', 'i64.div_u']:
        stack[top-2] = int(stack[top-2] / stack[top-1])
        top -= 1
    
    elif op in ['i32.rem_s', 'i64.rem_s']:
        pass

    elif op in ['i32.rem_u', 'i64.rem_u']:
        stack[top-2] -= stack[top-1] * int(stack[top-2] / stack[top-1])
        top -= 1

    elif op in ['i32.and', 'i64.and']:
        stack[top-2] &= stack[top-1]
        top -= 1

    elif op in ['i32.or', 'i64.or']:
        stack[top-2] |= stack[top-1]
        top-= 1

    elif op in ['i32.xor', 'i64.xor']:
        stack[top-2] ^= stack[top-1]
        top -= 1

    elif op in ['i32.shl', 'i64.shl']:
        modulo = 2**int(op[1:3])
        shift_len = stack[top-1] % modulo
        stack[top-2] = (stack[top-2] << shift_len) % modulo
        top -= 1

    elif op in ['i32.shr_s', 'i64.shr_s']:
        modulo = 2**int(op[1:3])
        shift_len = stack[top-1] % modulo
        stack[top-2] = stack[top-2] >> shift_len
        top -= 1

    elif op in ['i32.shr_u', 'i64.shr_u']:
        modulo = 2**int(op[1:3])
        shift_len = stack[top-1] % modulo
        binary_num = bin(stack[top-2])[2:]
        binary_num = ('0'*shift_len + binary_num)[:len(binary_num)]
        stack[top-2] = int(binary_num, base=2)
        top -= 1

    elif op in ['i32.rotl', 'i64.rotl']:
        int_bits = int(op[1:3])
        shift_len = stack[top-1] % (2**int_bits)
        stack[top-2] = (stack[top-2] << shift_len) | (stack[top-2] >> (int_bits-shift_len))
        if int_bits == 32:  # Python has not bits limit
            stack[top-2] &= 0xFFFFFFFF
        else:
            stack[top-2] &= 0xFFFFFFFFFFFFFFFF
        top -= 1

    elif op in ['i32.rotr', 'i64.rotr']:
        int_bits = int(op[1:3])
        shift_len = stack[top-1] % (2**int_bits)
        stack[top-2] = (stack[top-2] >> shift_len) | (stack[top-2] << (int_bits-shift_len))
        if int_bits == 32:
            stack[top-2] &= 0xFFFFFFFF
        else:
            stack[top-2] &= 0xFFFFFFFFFFFFFFFF
        top -= 1

    elif op in ['f32.abs', 'f64.abs']:
        stack[top-1] = abs(stack[top-1])

    elif op in ['f32.neg', 'f64.neg']:
        stack[top-1] = -stack[top-1]

    elif op in ['f32.ceil', 'f64.ceil']:
        stack[top-1] = ceil(stack[top-1])

    elif op in ['f32.floor', 'f64.floor']:
        stack[top-1] = floor(stack[top-1])

    elif op in ['f32.trunc', 'f64.trunc']:
        