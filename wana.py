#!/usr/bin/python3

import argparse
import os

def main(args):

    # Set the commands for contract analysis
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--wast', type=str, help='Check WAST format file')
    parser.add_argument('-m', '--wasm', type=str, help='Check WASM format file')

    args = parser.parse_args(args)

    # Compile the .wast file to .wasm file
    if args.wast:
        wasm_code_path = './tmp/temp.wasm'
        compile_contract(args.wast, wasm_code_path)
        if not os.path.isfile(wasm_code_path):
            print('Contract %s does not exist!' % args.wasm[1])
    if args.wast or args.wasm:
        print('Successfully input command...')
    
    # Create the private blockchain
    create_private_chain('emptychain', GlobalVar.eos_account)

    # Deploy the contract
    if args.wast:
        contract_address = deploy_contract(wasm_code_path, GlobalVar.eos_account)
    else:
        contract_address = deploy_contract(args.wasm, GlobalVar.eos_account)
    
    if contract_address is None:
        print('Failed to deploy the contract!')
        return
