import ethereum_data_root as e
import sys
import os

if __name__ == '__main__':
    L = int(sys.argv[1])
    data = os.urandom(L)
    print(e.mk_data_root(data))
