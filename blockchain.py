import os
import subprocess

def create_private_chain(chain, eosbase):
    
    devnull = open(os.devnull, 'w')

    if chain != 'emptychain':

        # Remove the previous blockchain
        prev = subprocess.Popen(['rm', '-rf', './blockchains/'+chain])
        prev.wait()

        # Init new blockchain
        