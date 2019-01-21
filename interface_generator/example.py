import interface_generator as ig
import json
interface = ig.generate_interface(
    '0xde0b295669a9fd93d5f28d9ec85e40f4cb697bae',
    json.load(open('abi.json')),
    json.load(open('instructions.json'))
)
open('out.html', 'w+').write(interface)

