#!/usr/bin/python3

# import argparse
# import os

# def main(args):

#     # Set the commands for contract analysis
#     parser = argparse.ArgumentParser()
#     parser.add_argument('-t', '--wast', type=str, help='Check WAST format file')
#     parser.add_argument('-m', '--wasm', type=str, help='Check WASM format file')

#     args = parser.parse_args(args)

#     # Compile the .wast file to .wasm file
#     if args.wast:
#         wasm_code_path = './tmp/temp.wasm'
#         compile_contract(args.wast, wasm_code_path)
#         if not os.path.isfile(wasm_code_path):
#             print('Contract %s does not exist!' % args.wasm[1])
#     if args.wast or args.wasm:
#         print('Successfully input command...')
    
#     # Create the private blockchain
#     create_private_chain('emptychain', GlobalVar.eos_account)

#     # Deploy the contract
#     if args.wast:
#         contract_address = deploy_contract(wasm_code_path, GlobalVar.eos_account)
#     else:
#         contract_address = deploy_contract(args.wasm, GlobalVar.eos_account)
    
#     if contract_address is None:
#         print('Failed to deploy the contract!')
#         return

import argparse
import os
import typing

from wana import convention
from wana import execution
from wana import log
from wana import runtime_structure

class Runtime:
    # A webassembly runtime manages Store, Stack, and other structures.
    # They forming the WebAssembly abstract.

    def __init__(self, module: runtime_structure.Module, imps: typing.Dict = None):
        self.module = module
        self.module_instance = execution.ModuleInstance()
        self.store = execution.Store()

        imps = imps if imps else {}
        externvals = []
        for e in self.module.imports:
            if e.module not in imps or e.name not in imps[e.module]:
                raise Exception(f'Global import {e.module}.{e.name} not found!')
            if e.kind == convention.extern_func:
                a = execution.HostFunc(self.module.types[e.desc], imps[e.module][e.name])
                self.store.funcs.append(a)
                externvals.append(execution.ExternValue(e.kind, len(self.store.funcs) - 1))
                continue
            if e.kind == convention.extern_table:
                a = imps[e.module][e.name]
                self.store.tables.append(a)
                externvals.append(execution.ExternValue(e.kind, len(self.store.tables) - 1))
                continue
            if e.kind == convention.extern_mem:
                a = imps[e.module][e.name]
                self.store.mems.append(a)
                externvals.append(execution.ExternValue(e.kind, len(self.store.mems) - 1))
                continue
            if e.kind == convention.extern_global:
                a = exection.GlobalInstance(execution.Value(e.desc.valtype, imps[e.module][e.name]), e.desc.mut)
                self.store.globals.append(a)
                externvals.append(execution.ExternValue(e.kind, len(self.store.globals) - 1))
                continue
        self.module_instance.instantiate(self.module, self.store, externvals)

    def func_addr(self, name: str):
        # Get a function address denoted by the function name.
        for e in self.module_instance.exports:
            if e. name == name and e.value.extern_type == convention.extern_func:
                return e.value.address
        raise Exception('Function not found!')

    def exec(self, name: str, args: typing.List):
        # Invoke a function denoted by the function address with the provided arguments.
        func_addr = self.func_addr(name)
        func = self.store.funcs[self.module_instance.funcaddrs[func_addr]]
        # Mapping check for Python valtype to WebAssembly valtype.
        for i, e in enumerate(func.functype.args):
            if e in [convention.i32, convention.i64]:
                assert isinstance(args[i], int)
            if e in [convention.f32, convention.f64]:
                assert isinstance(args[i], float)
            args[i] = execution.Value(e, args[i])
        stack = execution.Stack()
        stack.ext(args)
        log.debugln(f'Running function {name}({", ".join([str(e) for e in args])}):')
        r = execution.call(self.module_instance, func_addr, self.store, stack)
        if r:
            return r[0].n
        return None

# These code are the API for using.
# This is to execute the analysis automatically.

def on_debug():
    log.lel = 1

def load(name: str, imps: typing.Dict = None) -> Runtime:
    # Generate a runtime by loading a file from disk.
    with op(name, 'rb') as f:
        module = runtime_structure.Module.from_reader(f)
        return Runtime(module, imps)

def main():

    # Set the commands for contract analysis
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--wast', type=str, help='Check WAST format file')
    parser.add_argument('-m', '--wasm', type=str, help='Check WASM format file')
    parser.add_argument('-e', '--execute', type=str, nargs='*', help='Execute a smart contract')

    args = parser.parse_args()

    if args.execute:
        vm = load(args.execute[0], args.execute[1])

if __name__ == '__main__':
    Memory = execution.MemoryInstance
    Value = execution.Value
    Table = execution.TableInstance
    Limits = runtime_structure.Limits
    main()