macro calldatachar($x):
    div(calldataload($x), 2**248)

macro calldatabytes_as_int($x, $b):
    div(calldataload($x), 256**(32-$b))


c0 = calldatachar(0)
if c0 < 192:
    ~invalid()
elif c0 < 248:
    pos = 1
    L = c0 - 192
else:
    pos = 1 + (c0 - 247)
    if calldatachar(1) == 0:
        ~invalid()
    L = calldatabytes_as_int(1, c0 - 247)
if pos + L != ~calldatasize():
    ~invalid()
startpos = pos

lastpos = 0
while pos < ~calldatasize():
    c = calldatachar(pos)
    if c < 128:
        lastpos = pos
        pos += 1
    elif c < 184:
        L = c - 128
        lastpos = pos
        pos += L + 1
    elif c < 192:
        if calldatachar(pos + 1) == 0:
            ~invalid()
        L = calldatabytes_as_int(pos + 1, c - 183)
        lastpos = pos
        pos += L + (c - 183) + 1
    else:
        ~invalid()


newlen = lastpos - startpos
newmemindex = 1000
if newlen < 56:
    ~mstore8(newmemindex, 192 + newlen)
    ~calldatacopy(newmemindex + 1, startpos, newlen)
    return(~sha3(newmemindex, 1 + newlen))
else:
    _log = 0
    _newlen = newlen
    while _newlen:
        _log += 1
        _newlen = ~div(_newlen, 256)
    ~mstore8(newmemindex, 247 + _log)
    ~mstore(newmemindex + 1, newlen * (256 ** (32 - _log)))
    ~calldatacopy(newmemindex + 1 + _log, startpos, newlen)
    return(~sha3(newmemindex, 1 + _log + newlen))
