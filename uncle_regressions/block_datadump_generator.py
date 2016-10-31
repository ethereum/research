import json
from ethereum.block import BlockHeader
import rlp
try:
    data = json.load(open('progress.json'))
    blknum = data['blknum']
    pos = data['pos']
except:
    blknum, pos = 0, 0
f = open('geth-2283415.dump')
outdata = []
while 1:
    f.seek(pos)
    prefix = f.read(10)
    _typ, _len, _pos = rlp.codec.consume_length_prefix(prefix, 0)
    blkdata = prefix + f.read(_pos + _len - 10)
    header = rlp.decode(rlp.descend(blkdata, 0), BlockHeader)
    txcount = len(rlp.decode(rlp.descend(blkdata, 1)))
    uncles = [BlockHeader.deserialize(x) for x in rlp.decode(rlp.descend(blkdata, 2))]
    outdata.append([header.number, len(uncles), sum([4.375 - 0.625 * (header.number - u.number) for u in uncles]), sum([u.gas_used for u in uncles]), txcount, header.gas_used, _len + _pos, blkdata.count('\x00')])
    print outdata[-1]
    pos += _pos + _len
