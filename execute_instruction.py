from instructions import *
from utils import *
from math import *
import struct  # The "struct" package could interpret binary format.
from ctypes import *

# Executing the singal instruction
def execute_one_instruction(instruction, stack, locals_value, globals_value, memory, elements=None):
    op = opcode[instruction[0]]
    top = len(stack)

    if op == 'unreachable':
        raise Exception("Encounter the 'unreachable'('trap')!")

    elif op == 'nop':
        pass  # The 'nop' opcode means 'Do nothing'.

    elif op == 'loop':
        pass

    elif op == 'if':
        pass
    
    elif op == 'else':
        pass

    elif op == 'end':
        pass

    elif op == 'br':
        targer_address = stack[top-1]

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
        top -= 1

    elif op == 'select':
        first = stack[top-1]
        second = stack[top-2]
        third = stack[top-3]
        if is_all_real(first, second, third):
            if first == 0:
                computed = second
            else:
                computed = third
        else:
            first = to_symbolic(first)
            second = to_symbolic(second)
            third = to_symbolic(third)

            computed = If(first == 0, second, third)
        
        stack[top-3] = simplify(computed) if is_expr(computed) else computed
        top -= 2


    elif op == 'get_local':
        locals_pos = instruction[1]
        stack.append(locals_value[locals_pos])
        top += 1

    elif op == 'set_local':
        locals_pos = instruction[1]
        locals_value[locals_pos] = stack[top-1]
        top -= 1

    elif op == 'tee_local':
        locals_pos = instruction[1]
        locals_value[locals_pos] = stack[top-1]

    elif op == 'get_global':
        globals_pos = instruction[1]
        stack.append(globals_value[globals_pos])
        top += 1

    elif op == 'set_global':
        globals_pos = instruction[1]
        globals_value[globals_pos] = stack[top-1]
        top -= 1

    elif op == 'i32.load':
        align = instruction[1]
        offset = instruction[2]

        if memory[0] and 2**align <= 4:
            pass

    elif op == 'i64.load':
        align = instruction[1]
        offset = instruction[2]
        if memory[0] and 2**align <= 8:
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
        i32_value = instruction[1]
        stack.append(i32_value)
        top += 1

    elif op == 'i64.const':
        i64_value = instruction[1]
        stack.append(i64_value)
        top += 1

    elif op == 'f32.const':
        f32_value = instruction[1]
        stack.append(f32_value)
        top += 1

    elif op == 'f64.const':
        f64_value = instruction[1]
        stack.append(f64_value)
        top += 1

    elif op in ['i32.eqz', 'i64.eqz']:
        first = stack[top-1]
        if is_real(first):
            if first == 0:
                computed = 1
            else:
                computed = 0
        else:
            computed = If(first == 0, BitVecVal(1, int(op[1:3])), BitVecVal(0, int(op[1:3])))
        
        stack[top-1] = simplify(computed) if is_expr(computed) else computed

    elif op in ['i32.eq', 'i64.eq']:
        first = stack[top-1]
        second = stack[top-2]
        if is_all_real(first, second):
            if first == second:
                computed = 1
            else:
                computed = 0
        else:
            first = to_symbolic(first, int(op[1:3]))
            second = to_symbolic(second, int(op[1:3]))
            computed = If( eq(first, second), BitVecVal(1, int(op[1:3])), BitVecVal(0, int(op[1:3])))
        
        stack[top-2] = simplify(computed) if is_expr(computed) else computed
        top -= 1

    elif op in ['f32.eq', 'f64.eq']:
        pass

    elif op in ['i32.ne', 'i64.ne']:
        first = stack[top-1]
        second = stack[top-2]
        if is_all_real(first, second):
            if first != second:
                computed = 1
            else:
                computed = 0
        else:
            first = to_symbolic(first, int(op[1:3]))
            second = to_symbolic(second, int(op[1:3]))
            computed = If( eq(first, second), BitVecVal(0, int(op[1:3])), BitVecVal(1, int(op[1:3])))
        
        stack[top-2] = simplify(computed) if is_expr(computed) else computed
        top -= 1
    
    elif op in ['f32.ne', 'f64.ne']:
        pass

    elif op in ['i32.lt_u', 'i64.lt_u']:
        first = stack[top-1]
        second = stack[top-2]
        if is_all_real(first, second):
            if first < second:
                computed = 1
            else:
                computed = 0
        else:
            computed = If(ULT(first, second), BitVecVal(1, int(op[1:3])), BitVecVal(0, Int(op[1:3])))
        
        stack[top-2] = simplify(computed) if is_expr(computed) else computed
        top -= 1

    elif op in ['i32.lt_s', 'i64.lt_s']:
        first = stack[top-1]
        second = stack[top-2]
        if is_all_real(first, second):
            if first < second:
                computed = 1
            else:
                computed = 0
        else:
            computed = If(first < second, BitVecVal(1, int(op[1:3])), BitVecVal(0, Int(op[1:3])))
        
        stack[top-2] = simplify(computed) if is_expr(computed) else computed
        top -= 1

    elif op in ['f32.lt', 'f64.lt']:
        pass

    elif op in ['i32.gt_u', 'i64.gt_u']:
        first = stack[top-1]
        second = stack[top-2]
        if is_all_real(first, second):
            if first < second:
                computed = 1
            else:
                computed = 0
        else:
            computed = If(UGT(first, second), BitVecVal(1, int(op[1:3])), BitVecVal(0, Int(op[1:3])))
        
        stack[top-2] = simplify(computed) if is_expr(computed) else computed
        top -= 1

    elif op in ['i32.gt_s', 'i64.gt_s']:
        first = stack[top-1]
        second = stack[top-2]
        if is_all_real(first, second):
            if first < second:
                computed = 1
            else:
                computed = 0
        else:
            computed = If(first > second, BitVecVal(1, int(op[1:3])), BitVecVal(0, Int(op[1:3])))
        
        stack[top-2] = simplify(computed) if is_expr(computed) else computed
        top -= 1

    elif op in ['f32.gt', 'f64.gt']:
        pass

    elif op in ['i32.le_u', 'i64.le_u']:
        first = stack[top-1]
        second = stack[top-2]
        if is_all_real(first, second):
            if first < second:
                computed = 1
            else:
                computed = 0
        else:
            computed = If(ULE(first, second), BitVecVal(1, int(op[1:3])), BitVecVal(0, Int(op[1:3])))
        
        stack[top-2] = simplify(computed) if is_expr(computed) else computed
        top -= 1

    elif op in ['i32.le_s', 'i64.le_s']:
                first = stack[top-1]
        second = stack[top-2]
        if is_all_real(first, second):
            if first < second:
                computed = 1
            else:
                computed = 0
        else:
            computed = If(first <= second, BitVecVal(1, int(op[1:3])), BitVecVal(0, Int(op[1:3])))
        
        stack[top-2] = simplify(computed) if is_expr(computed) else computed
        top -= 1

    elif op in ['f32.le', 'f64.le']:
        pass

    elif op in ['i32.ge_u', 'i64.ge_u']:
        first = stack[top-1]
        second = stack[top-2]
        if is_all_real(first, second):
            if first < second:
                computed = 1
            else:
                computed = 0
        else:
            computed = If(UGE(first, second), BitVecVal(1, int(op[1:3])), BitVecVal(0, Int(op[1:3])))
        
        stack[top-2] = simplify(computed) if is_expr(computed) else computed
        top -= 1

    elif op in ['i32.ge_s',  'i64.ge_s']:
        first = stack[top-1]
        second = stack[top-2]
        if is_all_real(first, second):
            if first < second:
                computed = 1
            else:
                computed = 0
        else:
            computed = If(first >= second, BitVecVal(1, int(op[1:3])), BitVecVal(0, Int(op[1:3])))
        
        stack[top-2] = simplify(computed) if is_expr(computed) else computed
        top -= 1

    elif op in ['f32.ge', 'f64.ge']:
        pass
    
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
        first = stack[top-1]
        second = stack[top-2]
        if is_real(first) and is_symbolic(second):
            first = BitVecVal(first, int(op[1:3]))
            computed = first + second
        elif is_symbolic(first) and is_real(second):
            second = BitVecVal(second, int(op[1:3]))
            computed = first + second
        else:
            computed = (first + second) % (2 ** 32)

        stack[top-2] = simplify(computed) if is_expr(computed) else computed
        top -= 1

    elif op in ['i32.sub', 'i64.sub']:
        first = stack[top-1]
        second = stack[top-2]
        if is_real(first) and is_symbolic(second):
            first = BitVecVal(first, int(op[1:3]))
            computed = first - second
        elif is_symbolic(first) and is_real(second):
            second = BitVecVal(second, int(op[1:3]))
            computed = first - second
        else:
            computed = (first - second) % (2 ** int(op[1:3]))
        
        stack[top-2] = simplify(computed) if is_expr(computed) else computed
        top -= 1

    elif op in ['i32.mul', 'i64.mul']:
        first = stack[top-1]
        second = stack[top-2]
        if is_real(first) and is_symbolic(second):
            first = BitVecVal(first, int(op[1:3]))
        elif is_symbolic(first) and is_real(second):
            second = BitVecVal(second, int(op[1:3]))
        computed = first * second % int(op[1:3])

        stack[top-2] = simplify(computed) if is_expr(computed) else computed
        top -= 1

    elif op in ['i32.div_u', 'i64.div_u']:
        first = stack[top-1]
        second = stack[top-2]
        if is_all_real(first, second):
            if second == 0:
                computed = 0
                # TODO: Error Processing!
            else:
                computed = first / second
        else:
            first = BitVecVal(first, int(op[1:3])) if is_real(first) else first
            second = BitVecVal(second, int(op[1:3])) if is_real(second) else second
            solver.push()
            solver.add(Not(second==0))
            if check_sat(solver) == unsat:
                computed = 0
            else:
                computed = UDIV(first, second)
            solver.pop()
        stack[top-2] = simplify(computed) if is_expr(computed) else computed
    
    elif op in ['i32.div_s', 'i64.div_s']:
        first = stack[top-1]
        second = stack[top-2]
        if is_all_real(first, second):
            if second == 0:
                computed = 0
            elif first == -2**(int(op[1:3])-1) and second == -1:
                computed = -2**(int(op[1:3])-1)
            else:
                sign = -1 if (fitst / second) < 0 else 1
                computed = sign * (abs(first) / abs(second))
        else:
            first = BitVecVal(first, int(op[1:3])) if is_real(first) else first
            second = BitVecVal(second, int(op[1:3])) if is_real(second) else second
            solver.push()
            solver.add(Not(second==0))
            if check_sat(solver) == unsat:
                computed = 0
            else:
                solver.push()
                solver.add(Not(And(first == -2*(int(op[1:3])-1), second == -1)))
                if check_sat(solver) == unsat:
                    computed = -2**(int(op[1:3])-1)
                else:
                    solver.push()
                    solver.add(first / second < 0)
                    sign = -1 if check_sat(solver) == sat else 1
                    z3_abs = lambda x: If(x >= 0, x, -x)
                    first = z3_abs(first)
                    second = z3_abs(second)
                    computed = sign * (first / second)
                    solver.pop()
                solver.pop()
            solver.pop()

        stack[top-2] = simplify(computed) if is_expr(computed) else computed
        top -= 1

    elif op in ['i32.rem_s', 'i64.rem_s']:
        first = stack[top-1]
        second = stack[top-2]
        if is_all_real(first, second):
            if second == 0:
                computed = 0
            else:
                sign = -1 if first < 0 else 1
                computed = sign * (abs(first) % abs(second))

        else:
            first = BitVecVal(first, int(op[1:3])) if is_real(first) else first
            second = BitVecVal(second, int(op[1:3])) if is_real(second) else second
            
            solver.push()
            solver.add(Not(second == 0))
            if check_sat == unsat:
                computed = 0
            else:
                first = to_signed(first, int(op[1:3]))
                second = to_signed(second, int(op[1:3]))
                solver.push()
                solver.add(first < 0)
                sign = BitVecVal(-1, int(op[1:3])) if check_sat(solver) == sat else BitVecVal(1, int(op[1:3]))
                solver.pop()

                z3_abs = lambda x: If(x >= 0, x, -x)
                first = z3_abs(first)
                second = z3_abs(second)

                computed = sign * (first % second)
            solver.pop()

        stack[top-2] = simplify(computed) if is_expr(computed) else computed
        top -= 1

    elif op in ['i32.rem_u', 'i64.rem_u']:
        first = stack[top-1]
        second = stack[top-2]
        
        if is_all_real(first, second):
            if second == 0:
                computed = 0
            else:
                first = to_unsigned(first, int(op[1:3]))
                second = to_unsigned(second, int(op[1:3]))
                computed = first % second
        else:
            first = to_symbolic(first)
            second = to_symbolic(second)
            
            solver.push()
            solver.add(Not(second==0))
            if check_sat(solver) == unsat:
                computed = 0
            else:
                computed = URem(first, second)
            solver.pop()


        stack[top-2] = simplify(computed) if is_expr(computed) else computed
        top -= 1

    elif op in ['i32.and', 'i64.and']:
        first = stack[top-1]
        second = stack[top-2]
        computed = first & second
        stack[top-2] = simplify(computed) if is_expr(computed) else computed
        top -= 1

    elif op in ['i32.or', 'i64.or']:
        first = stack[top-1]
        second = stack[top-2]
        computed = first | second
        stack[top-2] = simplify(computed) if is_expr(computed) else computed
        top-= 1

    elif op in ['i32.xor', 'i64.xor']:
        first = stack[top-1]
        second = stack[top-2]
        computed = first ^ second
        stack[top-2] = simplify(computed) if is_expr(computed) else computed
        top -= 1

    elif op in ['i32.shl', 'i64.shl']: # waiting to fix-up
        first = stack[top-1]
        second = stack[top-2]
        modulo = int(op[1:3])

        if is_all_real(first, second):
            if first >= modulo or first < 0:
                computed = 0
            else:
                computed = second << (first % modulo)
        else:
            first = to_symbolic(first, modulo)
            second = to_symbolic(second, modulo)
            solver.push()
            solver.add(Not(Or(first >= modulo, first < 0)))
            if check_sat(solver) == unsat:
                computed = 0
            else:
                computed = second << (first % modulo)
            solver.pop()
        
        stack[top-2] = simplify(computed) if is_expr(computed) else computed
        top -= 1

    elif op in ['i32.shr_s', 'i64.shr_s']:
        first = stack[top-1]
        second = stack[top-2]
        modulo = int(op[1:3])
        
        if is_all_real(first, second):
            if first < 0:
                computed = 0
            else:
                computed = second >> (first % modulo)
        else:
            first = to_symbolic(first, modulo)
            second = to_symbolic(second, modulo)
            solver.push()
            solver.add(Not(first < 0))
            if check_sat(solver) == unsat:
                computed = 0
            else:
                computed = second >> (first % modulo)
            solver.pop()
        
        stack[top-2] = simplify(computed) if is_expr(computed) else computed
        top -= 1

    elif op in ['i32.shr_u', 'i64.shr_u']: # TODO: fix-up
        first = stack[top-1]
        second = stack[top-2]
        modulo = int(op[1:3])
        bit_len = int('F' * (modulo // 4), base=16)

        if is_all_real(first, second):
            if first < 0 or first > modulo:
                computed = 0
            elif first == 0:
                computed = stack[top-2]
            else:
                computed = (second & bit_len) >> (first % modulo)
        else:
            first = to_symbolic(first, modulo)
            second = to_symbolic(second, modulo)
            solver.push()
            solver.add(Not(first < 0))
            if check_sat(solver) == unsat:
                computed = 0
            else:
                computed = UShR(second, (first % modulo)) 
            solver.pop()

        stack[top-2] = simplify(computed) if is_expr(computed) else computed
        top -= 1

    elif op in ['i32.rotl', 'i64.rotl']: # TODO: fix-up
        first = stack[top-1]
        second = stack[top-2]
        modulo = int(op[1:3])
        bit_len = int('F' * (modulo // 4), base=16)

        if is_all_real(first, second):
            move_len = first % modulo
            second &= bit_len
            computed = (second >> (modulo - move_len)) | (second << move_len)
        else:
            first = to_symbolic(first, modulo)
            second = to_symbolic(second, modulo)
            move_len = first % modulo
            second &= bit_len
            computed = (second >> (modulo - move_len)) | (second << move_len)
        
        stack[top-2] = simplify(computed) if is_expr(computed) else computed
        top -= 1

    elif op in ['i32.rotr', 'i64.rotr']: # TODO: fix-up
        first = stack[top-1]
        second = stack[top-2]
        modulo = int(op[1:3])
        bit_len = int('F' * (modulo // 4), base=16)

        if is_all_real(first, second):
            move_len = first % modulo
            second &= bit_len
            computed = (second << (modulo - move_len)) | (second >> move_len)
        else:
            first = to_symbolic(first, modulo)
            second = to_symbolic(second, modulo)
            move_len = first % modulo
            second &= bit_len
            computed = (second << (modulo - move_len)) | (second >> move_len)
        
        stack[top-2] = simplify(computed) if is_expr(computed) else computed
        top -= 1

    # TODO: float number
    elif op in ['f32.abs', 'f64.abs']:
        first = stack[top-1]

        if is_real(first):
            computed = abs(first)
        else:
            z3_abs = lambda x: If(x >= 0, x, -x)
            computed = z3_abs(first)
        
        computed = simplify(computed) if is_expr(computed) else computed

    elif op in ['f32.neg', 'f64.neg']:
        stack[top-1] = -stack[top-1]

    elif op in ['f32.ceil', 'f64.ceil']:
        stack[top-1] = float(ceil(stack[top-1]))

    elif op in ['f32.floor', 'f64.floor']:
        stack[top-1] = float(floor(stack[top-1]))

    elif op in ['f32.trunc', 'f64.trunc']:
        stack[top-1] = float(trunc(stack[top-1]))

    elif op in ['f32.nearest', 'f64.nearest']:
        stack[top-1] = float(round(stack[top-1]))

    elif op in ['f32.sqrt', 'f64.sqrt']:
        stack[top-1] = sqrt(stack[top-1])

    elif op in ['f32.add', 'f64.add']:
        stack[top-2] = stack[top-2] + stack[top-1]

    elif op in ['f32.sub', 'f64.sub']:
        stack[top-2] = stack[top-2] - stack[top-1]
        top -= 1

    elif op in ['f32.mul', 'f64.mul']:
        stack[top-2] = stack[top-2] * stack[top-1]
        top -= 1

    elif op in ['f32.div', 'f64.div']:
        stack[top-2] = stack[top-2] / stack[top-1]
        top -= 1

    elif op in ['f32.min', 'f64.min']:
        stack[top-2] = min(stack[top-2], stack[top-1])
        top -= 1
    
    elif op in ['f32.max', 'f64.max']:
        stack[top-2] = max(stack[top-2], stack[top-1])
        top -= 1

    elif op in ['f32.copysign', 'f64.copysign']:
        stack[top-2] = copysign(stack[top-2], stack[top-1])
        top -= 1

    # END: Float

    elif op in ['i32.wrap_i64']:
        first = stack[top-1]
        
        if is_real(first):
            sign = -1 if first < 0 else 1
            computed = sign * (abs(first) % 2**32)
        else:
            solver.push()
            solver.add(first < 0)
            sign = BitVecVal(-1, 32) if check_sat(solver) == sat \
                else BitVecVal(1, 32)
            solver.pop()

            z3_abs = lambda x: If(x >= 0, x, -x)
            first = z3_abs(first)
            computed = sign * (first % 2**32)
        
        stack[top-1] = simplify(computed) if is_expr(computed) else computed

    # TODO: Float-related instructon.
    elif op in ['i32.trunc_s/f32', 'i32.trunc_u/f32', 'i32.trunc_s/f64', 'i32.trunc_u/f64']:
        stack[top-1] = int(stack[top-1])

    elif op in ['i64.extend_i32_s']:
        first = stack[top-1]
        if is_real(first):
            computed = first
        else:
            computed = SignExt(32, first)
        
        stack[top-1] = simplify(computed) if is_expr(computed) else computed
    
    elif op in ['i64.extend_i32_u']:
        first = stack[top-1]
        if is_real(first):
            computed = first & 0xFFFFFFFF
        else:
            computed = ZeroExt(32, first)
        
        stack[top-1] = simplify(computed) if is_expr(computed) else computed

    elif op in ['i64.trunc_s/f32', 'i64.trunc_u/f32', 'i64.trunc_s/f64', 'i64.trunc_u/f64']:  # TODO : Implement the function
        stack[top-1] = int(stack[top-1])
    
    elif op in ['f32.convert_s/i32', 'f32.convert_u/i32', 'f32.convert_s/i64', 'f32.convert_u/i64']:
        stack[top-1] = float(stack[top-1])

    elif op in ['f32.demote/f64']:
        pass

    elif op in ['f64.convert_s/i32', 'f64.convert_u/i32', 'f64.convert_s/i64', 'f64.convert_u/i64']:
        stack[top-1] = float(stack[top-1])

    elif op in ['f64.promote/f32']:
        pass

    elif op in ['i32.reinterpret/f32']:
        stack[top-1] = struct.unpack('l', struct.pack('f', stack[top-1]))[0]

    elif op in ['i64.reinterpret/f64']:
        stack[top-1] = struct.unpack('L', stack.pack('d', stack[top-1]))[0]

    elif op in ['f32.reinterpret/i32']:
        stack[top-1] = struct.unpack('f', struct.pack('l', stack[top-1]))[0]

    elif op in ['f64.reinterpret/i64']:
        stack[top-1] = struct.unpack('d', struct.pack('L', stack[top-1]))

    else:
        log.debug('Unknown instruction: ' + opcode)
        raise Exception('Unknown instruction: ' + opcode)

    
    # Correct the size of stack
    stack = stack[:top]

