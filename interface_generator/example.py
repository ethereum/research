import interface_generator as ig
import json
import sys

file2json = lambda x: json.load(open(x))

examples = file2json('examples/addresses.json')

if len(sys.argv) == 1:
    print("Available examples: " + ", ".join(examples.keys()))
elif sys.argv[1] in examples:
    address = examples[sys.argv[1]]
    abi = file2json('examples/%s_abi.json' % sys.argv[1])
    instructions = file2json('examples/%s_instructions.json' % sys.argv[1])
    interface = ig.generate_interface(address, abi, instructions)
    open('examples/%s_out.html' % sys.argv[1], 'w+').write(interface)
    print("Outputted file to examples/%s_out.html" % sys.argv[1])
else:
    print("Example %s not found" % sys.argv[1])
