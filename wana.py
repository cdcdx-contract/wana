#!/usr/bin/python3

import argparse
import os
import typing
import z3

import convention
import execute_instruction
import log
import runtime_structure
from global_variables import path_conditions_and_results

class Runtime:
    # A webassembly runtime manages Store, Stack, and other structures.
    # They forming the WebAssembly abstract.

    def __init__(self, module: runtime_structure.Module, imps: typing.Dict = None):
        self.module = module
        self.module_instance = execute_instruction.ModuleInstance()
        self.store = execute_instruction.Store()

        imps = imps if imps else {}
        externvals = []
        for e in self.module.imports:
            if e.module not in imps or e.name not in imps[e.module]:
                raise Exception(f'Global import {e.module}.{e.name} not found!')
            if e.kind == convention.extern_func:
                a = execute_instruction.HostFunc(self.module.types[e.desc], imps[e.module][e.name])
                self.store.funcs.append(a)
                externvals.append(execute_instruction.ExternValue(e.kind, len(self.store.funcs) - 1))
                continue
            if e.kind == convention.extern_table:
                a = imps[e.module][e.name]
                self.store.tables.append(a)
                externvals.append(execute_instruction.ExternValue(e.kind, len(self.store.tables) - 1))
                continue
            if e.kind == convention.extern_mem:
                a = imps[e.module][e.name]
                self.store.mems.append(a)
                externvals.append(execute_instruction.ExternValue(e.kind, len(self.store.mems) - 1))
                continue
            if e.kind == convention.extern_global:
                a = execute_instruction.GlobalInstance(execute_instruction.Value(e.desc.valtype, imps[e.module][e.name]), e.desc.mut)
                self.store.globals.append(a)
                externvals.append(execute_instruction.ExternValue(e.kind, len(self.store.globals) - 1))
                continue
        self.module_instance.instantiate(self.module, self.store, externvals)

    def func_addr(self, name: str):
        # Get a function address denoted by the function name.
        for e in self.module_instance.exports:
            if e.name == name and e.value.extern_type == convention.extern_func:
                return e.value.addr
        raise Exception('Function not found!')

    def exec(self, name: str, args: typing.List):
        # Invoke a function denoted by the function address with the provided arguments.
        func_addr = self.func_addr(name)
        func = self.store.funcs[self.module_instance.funcaddrs[func_addr]]
        # Mapping check for Python valtype to WebAssembly valtype.
        for i, e in enumerate(func.functype.args):
            if e in [convention.i32, convention.i64]:
                assert isinstance(args[i], int) or isinstance(args[i], z3.BitVecRef)
            
            # [TODO] Float type symbolic executing.
            if e in [convention.f32, convention.f64]:
                assert isinstance(args[i], float)

            args[i] = execute_instruction.Value(e, args[i])
        stack = execute_instruction.Stack()
        stack.ext(args)
        log.debugln(f'Running function {name}({", ".join([str(e) for e in args])}):')
        r = execute_instruction.call(self.module_instance, func_addr, self.store, stack)
        if r:
            return r
        return None

    def exec_by_address(self, address: int, args: typing.List):
        # Invoke a function denoted by the function address with the provided arguments.
        func = self.store.funcs[self.module_instance.funcaddrs[address]]
        # Mapping check for Python valtype to WebAssembly valtype.
        for i, e in enumerate(func.functype.args):
            if e in [convention.i32, convention.i64]:
                assert isinstance(args[i], int) or isinstance(args[i], z3.BitVecRef)
            
            # [TODO] Float type symbolic executing.
            if e in [convention.f32, convention.f64]:
                assert isinstance(args[i], float)

            args[i] = execute_instruction.Value(e, args[i])
        stack = execute_instruction.Stack()
        stack.ext(args)
        log.debugln(f'Running function address {address}({", ".join([str(e) for e in args])}):')
        r = execute_instruction.call(self.module_instance, address, self.store, stack)
        if r:
            return r
        return None

# These code are the API for using.
# This is to execute the analysis automatically.

def on_debug():
    log.lel = 1

def load(name: str, imps: typing.Dict = None) -> Runtime:
    # Generate a runtime by loading a file from disk.
    with open(name, 'rb') as f:
        module = runtime_structure.Module.from_reader(f)
        return Runtime(module, imps)

def main():
    # Set the commands for contract analysis
    parser = argparse.ArgumentParser()
    parser.add_argument('-wt', '--wast', type=str, help='Check WAST format file')
    parser.add_argument('-wm', '--wasm', type=str, help='Check WASM format file')
    parser.add_argument('-e', '--execute', type=str, nargs='*', help='Execute a smart contract')
    parser.add_argument('-v', '--version', action='version', version='wana version 0.1.0 - buaa-scse-les')
    parser.add_argument('-t', '--timeout', help='Timeout for analysis using z3 in ms.', action='store', type=int)
    args = parser.parse_args()

    # Execute all export functions of wasm
    if args.execute:
        vm = load(args.execute[0])
        for e in vm.module_instance.exports:
            if e.value.extern_type == convention.extern_func:
                print(vm.exec_by_address(e.value.addr, [z3.BitVec('x', 32), 32]))
                print(path_conditions_and_results)

if __name__ == '__main__':
    Ctx = execute_instruction.Ctx
    Memory = execute_instruction.MemoryInstance
    Value = execute_instruction.Value
    Table = execute_instruction.TableInstance
    Limits = runtime_structure.Limits
    main()