def compress_fri(prf):
    o = []
    oindex = {}
    def add_obj(x):
        if x in oindex:
            o.append(oindex[x].to_bytes(2, 'big'))
        else:
            o.append(x)
            oindex[x] = len(o)-1

    for root, yproofs in prf[:-1]:
        # print('Adding proof item, pos %d' % len(o))
        add_obj(b'----')
        add_obj(root)
        for yproof in yproofs:
            for branch in yproof:
                for p in branch:
                    add_obj(p)
                add_obj(b'++++')
            add_obj(b'====')

    # print('Adding final proof, pos %d' % len(o))
    add_obj(b'////')
    for x in prf[-1]:
        add_obj(x)
    assert decompress_fri(o) == prf
    return o

def decompress_fri(proof):
    def get_obj(pos):
        return proof[int.from_bytes(proof[pos], 'big')] if len(proof[pos]) == 2 else proof[pos]
    o = []
    pos = 0
    while proof[pos] != b'////':
        # print("Processing proof item", pos)
        assert get_obj(pos) == b'----'
        root = get_obj(pos + 1)
        pos += 2
        yproofs = []
        while get_obj(pos) not in (b'----', b'////'):
            yproof = []
            while get_obj(pos) != b'====':
                branch = []
                while get_obj(pos) != b'++++':
                    branch.append(get_obj(pos))
                    pos += 1
                yproof.append(branch)
                pos += 1
            yproofs.append(yproof)
            pos += 1
        o.append([root, yproofs])
    # print('Processing final proof, pos %d' % pos)
    pos += 1
    o.append([get_obj(x) for x in range(pos, len(proof))])
    return o

def compress_branches(branches):
    o = []
    oindex = {}
    def add_obj(x):
        if x in oindex:
            o.append(oindex[x].to_bytes(2, 'big'))
        else:
            o.append(x)
            oindex[x] = len(o)-1

    for branch in branches:
        for p in branch:
            add_obj(p)
        add_obj(b'----')
    assert decompress_branches(o) == branches
    return o

def decompress_branches(proof):
    def get_obj(pos):
        return proof[int.from_bytes(proof[pos], 'big')] if len(proof[pos]) == 2 else proof[pos]
    o = []
    pos = 0
    while pos < len(proof):
        branch = []
        while pos < len(proof) and get_obj(pos) != b'----':
            branch.append(get_obj(pos))
            pos += 1
        o.append(branch)
        pos += 1
    return o

def bin_length(c):
    return len(b''.join([(b'\xff' if len(x) == 32 else b'') + x for x in c]))
