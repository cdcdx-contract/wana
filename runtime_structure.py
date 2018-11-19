import struct

value_type = {0x7f: 'i32', 0x7e: 'i64', 0x7d: 'f32', 0x7c: 'f64'}

class Module(object):
    def __init__(self, bytecode):
        self.types = []
        self.funcs = []
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
        section_code, section_size, num = struct.unpack('b', bytecode[pos:pos+3])
        pos += 3
        for i in range(num):
            # func_type = bytecode[pos]  # reserved for future
            params_num = struct.unpack('b', bytecode[pos+1])[0]
            results_num = struct.unpack('b', bytecode[pos+1+params_num+1])[0]
            params_type = list(struct.unpack('b'*params_num, bytecode[pos+2:pos+2+params_num]))
            results_type = list(struct.unpack('b'*results_num, bytecode[pos+3+params_num:pos+3+params_num+results_num]))
            params_type = [value_type[params_type[j]] for j in range(len(params_type))]  # convert the format hex to str
            results_type = [value_type[results_type[j]] for j in range(len(results_type))]
            self.types.append(Type(params_type, results_type))
            pos += 3 + params_num + results_num

        # Construct section 'Import'
        section_code, section_size, num = struct.unpack('b', bytecode[pos:pos+3])
        pos += 3
        for i in range(num):



class Type(object):
    def __init__(self, params_type, results_type, func_type=None):
        self.params_type = params_type
        self.results_type = results_type
        self.func_type = func_type

class Import(object):
    def __init__(self, )