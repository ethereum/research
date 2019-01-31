from hashlib import blake2s

SSZ_CHUNK_SIZE = 32

# Hash algorithm; change to whatever works
def hash(x): return blake2s(x).digest()

# Next power of 2 >= the input
def next_power_of_2(x):
    if x <= 2:
        return x
    elif x % 2 == 0:
        return 2 * next_power_of_2(x // 2)
    else:
        return 2 * next_power_of_2((x+1) // 2)

# Pad a value with zero-bytes to the given length
def zpad(x, n):
    return x + b'\x00' * (n - len(x))

class HashableList():

    def __init__(self, values):
        self.item_length = len(values[0])
        # Check type homogeneity
        for v in values:
            assert len(v) == self.item_length
        self.items_per_chunk = SSZ_CHUNK_SIZE // self.item_length

        # Build a list of chunks based on the number of items in the chunk. Note that
        # https://github.com/ethereum/eth2.0-specs/pull/538 is assumed to be in the spec here
        chunks = [
            zpad(b''.join(values[i:i+self.items_per_chunk]), SSZ_CHUNK_SIZE)
            for i in range(0, len(values), self.items_per_chunk)
        ]
        self.length = len(values)
        # Fill chunks up to the next power of 2
        chunks += [b'\x00' * SSZ_CHUNK_SIZE] * (next_power_of_2(len(chunks)) - len(chunks))

        # Calculate and store Merkle tree
        self.tree = [b''] * len(chunks) + chunks
        for i in range(len(self.tree)//2-1, 0, -1):
            self.tree[i] = hash(self.tree[i*2] + self.tree[i*2+1])

    # Array getter. Retrieves the correct part of the correct item in self.tree
    def __getitem__(self, i):
        item = self.tree[len(self.tree)//2 + i // self.items_per_chunk]
        return item[self.item_length * (i % self.items_per_chunk):][:self.item_length]

    # Array setter. Retrieves the correct part of the correct item in self.tree, and
    # recalculates the Merkle tree branches
    def __setitem__(self, i, value):
        assert len(value) == self.item_length
        assert i < self.length
        poz = len(self.tree)//2 + i // self.items_per_chunk
        item = self.tree[poz]
        startpos = self.item_length * (i % self.items_per_chunk)
        new_item = item[:startpos] + value + item[startpos+self.item_length:]
        # Recalculate Merkle tree
        while poz > 0:
            self.tree[poz] = new_item
            poz //= 2
            new_item = hash(self.tree[poz * 2] + self.tree[poz * 2 + 1])

    # Append to the list
    def append(self, value):
        # If the length of the chunks hits a power of two, expand self.tree
        poz = len(self.tree)//2 + self.length // self.items_per_chunk
        if poz == len(self.tree):
            print("boo")
            self.tree = [b''] * len(self.tree) + self.tree[len(self.tree)//2:] + [b'\x00' * SSZ_CHUNK_SIZE] * (len(self.tree) // 2)
            for i in range(len(self.tree)//2-1, 0, -1):
                self.tree[i] = hash(self.tree[i*2] + self.tree[i*2+1])
        # Add in the new item
        self.length += 1
        self[self.length-1] = value

    # Pop from the list
    def pop(self):
        # Remove the item. Note that if the length of the chunks goes below a power
        # of two, we do NOT remove the item (this is to prevent DoS attacks by
        # repeatedly expanding and contracting a list around a power of 2, causing
        # repeated O(n) resizes), instead we handle the size reduction in the root method
        self[self.length-1] = b'\x00' * self.item_length
        self.length -= 1

    def __len__(self):
        return self.length

    @property
    def root(self):
        # If the tree is more than 2x too big, then we simulate having a smaller tree
        # by returning the root of the left half (or the left half of the left half etc)
        # of the tree
        last_chunk_offset = (self.length-1) // self.items_per_chunk
        root_index = 1
        while last_chunk_offset < len(self.tree) // 4:
            last_chunk_offset *= 2
            root_index *= 2
        return hash(self.tree[root_index] + self.length.to_bytes(32, 'little'))
