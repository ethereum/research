from hashlib import sha256

def hash(x):
    return sha256(x).digest()

def merkelize(vals):
    assert len(vals) & (len(vals)-1) == 0
    o = [None] * len(vals) + [hash(x) for x in vals]
    for i in range(len(vals)-1, 0, -1):
        o[i] = hash(o[i*2] + o[i*2+1])
    return o

def get_root(tree):
    return tree[1]

def get_branch(tree, pos):
    offset_pos = pos + len(tree) // 2
    branch_length = len(tree).bit_length() - 2
    return [tree[(offset_pos >> i)^1] for i in range(branch_length)]

def verify_branch(root, pos, val, branch):
    x = hash(val)
    for b in branch:
        if pos & 1:
            x = hash(b + x)
        else:
            x = hash(x + b)
        pos //= 2
    return x == root
