import struct
from utils import *
from instructions import *

value_type = {0x7f: 'i32', 0x7e: 'i64', 0x7d: 'f32', 0x7c: 'f64'}
import_kind = {0x00: 'func', 0x01: 'table', 0x02: 'mem', 0x03: 'global'}
export_kind = {0x00: 'func', 0x01: 'table', 0x02: 'mem', 0x03: 'global'}
table_type = {0x70: 'anyfunc'}

class Module(object):
    def __init__(self, bytecode):
        self.types = []
        self.funcs = []
        self.codes = []
        self.tables = []
        self.mems = []
        self.globals = []
        self.exports = []
        self.imports = []

        pos = 0
        byte_len = len(bytecode)
        # Get magic number and version
        # magic_num = bytecode[0:4]
        # version = bytecode[4:8]
        pos += 8

        # Construct section 'Type'
        section_code = struct.unpack('b', bytecode[pos])[0][0]
        pos += 1

        section_size, byte_move = read_u32(bytecode, pos) # TODO: ERROR if return -1
        pos += byte_move

        num, byte_move = read_u32(bytecode[pos])
        pos += byte_move
    
        for i in range(num):
            # func_type = bytecode[pos]  # reserved for future
            params_num = struct.unpack('b', bytecode[pos+1])[0]
            results_num = struct.unpack('b', bytecode[pos+1+params_num+1])[0]
            params_type = list(struct.unpack('b'*params_num, bytecode[pos+2:pos+2+params_num]))
            results_type = list(struct.unpack('b'*results_num, bytecode[pos+3+params_num:pos+3+params_num+results_num]))
            params_type = [value_type[params_type[j]] for j in range(len(params_type))]  # convert the format hex to str
            results_type = [value_type[results_type[j]] for j in range(len(results_type))]
            pos += 3 + params_num + results_num
            self.types.append(Type(params_type, results_type))

        # Construct section 'Import'
        section_code = struct.unpack('b', bytecode[pos])[0]
        pos += 1
        
        section_size, byte_move = read_u32(bytecode, pos)
        pos += byte_move

        num, byte_move = read_u32(bytecode, pos)
        pos += byte_move

        for i in range(num):
            str_len, byte_move = read_u32(bytecode, pos)
            pos += byte_move
            module_name = bytecode[pos:pos+str_len].decode('utf8')
            pos += str_len

            str_len, byte_move = read_u32(bytecode, pos)
            pos += byte_move
            field_name = bytecode[pos:pos+str_len].decode('utf8')
            pos += str_len

            kind = import_kind[struct.unpack('b', bytecode[pos])[0]]
            pos += 1
            if kind == 'func':
                funcidx, byte_move = read_u32(bytecode, pos)
                pos += byte_move
            else:
                pass # TODO: table, mem, global

            self.imports.append(Import(kind, funcidx, [module_name, field_name]))

        # Construct section 'Function'
        section_code = struct.unpack('b', bytecode[pos])[0]
        pos += 1

        section_size, byte_move = read_u32(bytecode, pos)
        pos += byte_move

        num, byte_move = read_u32(bytecode, pos)
        pos += byte_move

        for i in range(num):
            funcidx, byte_move = read_u32(bytecode, pos)
            pos += byte_move
            self.funcs.append(funcidx)

        # Construct section 'Table'
        section_code = struct.unpack('b', bytecode[pos])[0]
        pos += 1

        section_size, byte_move = read_u32(bytecode, pos)
        pos += byte_move

        num, byte_move = read_u32(bytecode, pos)
        pos += byte_move

        for i in range(num):
            elemtype, flag = struct.unpack('bb', bytecode[pos:pos+2]) # elemtype is 0x70 at present
            pos += 2

            if flag == 0x00:
                min_limit, byte_move = read_u32(bytecode, pos)
                max_limit = None
                pos += byte_move
            elif flag == 0x01:
                min_limit, byte_move = read_u32(bytecode, pos)
                pos += byte_move
                max_limit, byte_move = read_u32(bytecode, pos)
                pos += byte_move
            else:
                pass  # TODO: Error processing

            self.tables.append(Table(elemtype, [min_limit, max_limit]))

        # Construct section 'Memory'
        section_code = struct.unpack('b', bytecode[pos])[0]
        pos += 1

        section_size, byte_move = read_u32(bytecode, pos)
        pos += byte_move

        num, byte_move = read_u32(bytecode, pos)
        pos += byte_move

        for i in range(num):
            flag = struct.unpack('b', bytecode[pos])[0]
            pos += 1

            if flag == 0x00:
                min_limit, byte_move = read_u32(bytecode, pos)
                max_limit = None
                pos += byte_move
            elif flag == 0x01:
                min_limit, byte_move = read_u32(bytecode, pos)
                pos += byte_move
                max_limit, byte_move = read_u32(bytecode, pos)
                pos += byte_move
            else:
                pass

        self.mems.append([min_limit, max_limit])

        # Construct section 'Export'
        section_code = struct.unpack('b', bytecode[pos])[0]
        pos += 1

        section_size, byte_move = read_u32(bytecode, pos)
        pos += byte_move

        num, byte_move = read_u32(bytecode, pos)
        pos += byte_move

        for i in range(num):
            str_len, byte_move = read_u32(bytecode, pos)
            pos += byte_move
            name = bytecode[pos:pos+str_len].decode('utf8')
            pos += str_len

            kind = export_kind[struct.unpack('b', bytecode[pos])[0]]
            pos += 1

            sigidx, byte_move = read_u32(bytecode, pos)
            pos += byte_move
            
            self.exports.append(Export(kind, sigidx, name))
            
        # Construct section 'Elem'
        section_code = struct.unpack('b', bytecode[pos])[0]
        pos += 1

        section_size, byte_move = read_u32(bytecode, pos)
        pos += byte_move

        num, byte_code = read_u32(bytecode, pos)
        pos += byte_move

        for i in range(num):
            tableidx, byte_move = read_u32(bytecode, pos)
            pos += byte_move

            op = opcode[struct.unpack('B', bytecode[pos])]
            pos += 1
            if op in operand_u32:
                offset, byte_move = read_u32(bytecode, pos)
            else:
                offset, byte_move = read_u64(bytecode, pos)
            pos += byte_move


class Type(object):
    def __init__(self, params_type, results_type, func_type=None):
        self.params_type = params_type
        self.results_type = results_type
        self.func_type = func_type



class Import(object):
    def __init__(self, kind, sigidx=None, name=None):
        self.kind = kind
        self.sigidx = sigidx
        self.name = name

class Table(object):
    def __init__(self, elemtype, limit):
        self.elemtype = elemtype
        self.limit = limit

class Memory(object):
    def __init__(self, limit):
        self.limit = limit
        self.space = [] # Initial empty space

class Export(object):
    def __init__(self， kind, sigidx=None, name=None):
        self.kind = kind
        self.sigidx = sigidx
        self.name = name

class Elem(object):
    def __init__(self, tableidx, expr=None, funcidx=None):
        self.tableidx = tableidx
        self.expr = expr
        self.funcidx = funcidx

class Code(object):
    def __init__(self, ):