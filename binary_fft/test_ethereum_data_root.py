import ethereum_data_root as e
import sys
import os
from hashlib import sha256
datastream = b''.join([sha256(bytes([i])).digest() for i in range(256)])

if __name__ == '__main__':
    L = int(sys.argv[1])
    data = (datastream * (L // len(datastream) + 1))[:L]
    print(e.mk_data_root(data))
