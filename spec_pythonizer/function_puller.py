import sys

code_lines = []

for i in (1, 2, 3, 4, 8, 32, 48, 96):
    code_lines.append("def int_to_bytes%d(x): return x.to_bytes(%d, 'little')" % (i, i))
code_lines.append("SLOTS_PER_EPOCH = 64")  # stub, will get overwritten by real var
code_lines.append("def slot_to_epoch(x): return x // SLOTS_PER_EPOCH")

code_lines.append("""
from typing import (
    Any,
    Callable,
    List,
    NewType,
    Tuple,
)


Slot = NewType('Slot', int)  # uint64
Epoch = NewType('Epoch', int)  # uint64
Shard = NewType('Shard', int)  # uint64
ValidatorIndex = NewType('ValidatorIndex', int)  # uint64
Gwei = NewType('Gwei', int)  # uint64
Bytes32 = NewType('Bytes32', bytes)  # bytes32
BLSPubkey = NewType('BLSPubkey', bytes)  # bytes48
BLSSignature = NewType('BLSSignature', bytes)  # bytes96
Any = None
Store = None
""")

pulling_from = None
current_name = None
processing_typedef = False
for linenum, line in enumerate(open(sys.argv[1]).readlines()):
    line = line.rstrip()
    if pulling_from is None and len(line) > 0 and line[0] == '#' and line[-1] == '`':
        current_name = line[line[:-1].rfind('`')+1: -1]
    if line[:9] == '```python':
        assert pulling_from is None
        pulling_from = linenum + 1
    elif line[:3] == '```':
        if pulling_from is None:
            pulling_from = linenum
        else:
            if processing_typedef:
                assert code_lines[-1] == '}'
                code_lines[-1] = '})'
            pulling_from = None
            processing_typedef = False
    else:
        if pulling_from == linenum and line == '{':
            code_lines.append('%s = SSZType({' % current_name)
            processing_typedef = True
        elif pulling_from is not None:
            code_lines.append(line)
        elif pulling_from is None and len(line) > 0 and line[0] == '|':
            row = line[1:].split('|')
            if len(row) >= 2:
                for i in range(2):
                    row[i] = row[i].strip().strip('`')
                    if '`' in row[i]:
                        row[i] = row[i][:row[i].find('`')]
                eligible = True
                if row[0][0] not in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ_':
                    eligible = False
                for c in row[0]:
                    if c not in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ_0123456789':
                        eligible = False
                if eligible:
                    code_lines.append(row[0] + ' = ' + (row[1].replace('**TBD**', '0x1234567890123567890123456789012357890')))

print(open('minimal_ssz.py').read())
print(open('bls_stub.py').read())

for line in code_lines:
    print(line)

print(open('state_transition.py').read())
print(open('monkey_patches.py').read())
