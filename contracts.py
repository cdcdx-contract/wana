from subprocess import Popen, PIPE
import os.path

def compile_contract(src_path, dest_path):

    print('Compiling .wast format contract file %s ...' %s src_path)

    if (not os.path.isfile(src_path)):
        print('Contract file %s does not exist.' % src_path)
        return

    with open(src_path, 'r') as wast_file:
        code = wast_file.read()

    p = Popen(['wat2wasm', src_path, '-o', dest_path], stdin=PIPE, stdout=PIPE, strerr=PIPE)
    compile_ans = ''
    while p.poll() is None:
        line = p.stdout.readline()
        compile_ans += bytes.decode(line)
    if 'Error' in compile_ans or 'error' in compile_ans:
        print(compile_ans)
        print('Failed to compile the contract.')
        exit()
    p.wait()

    print('The contract has been compiled.')