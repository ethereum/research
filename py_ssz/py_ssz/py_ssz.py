from .utils import int_to_big_endian
from .serializers import Serializable, CountableList

def add_length_prefix(obj, is_list=False):
    assert len(obj) <= 8388607
    return bytes([(len(obj) >> 16) + 128 * is_list,
                  (len(obj) >> 8) % 256,
                   len(obj) % 256]) + obj

    
def encode(obj):
    if isinstance(obj, bytes):
        return add_length_prefix(obj)
    elif isinstance(obj, str):
        return add_length_prefix(obj.encode('utf-8'))
    elif isinstance(obj, int):
        assert obj >= 0
        return add_length_prefix(int_to_big_endian(obj))
    elif isinstance(obj, list):
        res = b''
        for o in obj:
            res += encode(o)
        return add_length_prefix(res, True)
    elif isinstance(obj, Serializable):
        return encode(obj._s(obj))

def decode_raw(obj, pos=0):
    startpos = pos
    L = ((obj[pos] % 128) << 16) + (obj[pos+1] << 8) + obj[pos+2]
    if obj[pos] < 128:
        assert len(obj) >= pos+3+L
        return obj[pos+3: pos+3+L], pos+3+L
    else:
        pos += 3
        o = []
        while pos < startpos+3+L:
            sub, pos = decode_raw(obj, pos)
            o.append(sub)
        assert pos == startpos+3+L
        return o, pos

def decode(obj, cls=None, *args, **kwargs):
    decoded, endpos = decode_raw(obj)
    assert endpos == len(obj)
    if cls is None:
        return decoded
    return cls._d(decoded, *args, **kwargs)
