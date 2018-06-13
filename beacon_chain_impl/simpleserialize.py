def serialize(val, typ=None):
    if typ is None and hasattr(val, 'fields'):
        typ = type(val)
    if typ in ('hash32', 'address'):
        assert len(val) == 20 if typ == 'address' else 32
        return val
    elif isinstance(typ, str) and typ[:3] == 'int':
        length = int(typ[3:])
        assert length % 8 == 0
        return val.to_bytes(length // 8, 'big')
    elif typ == 'bytes':
        return len(val).to_bytes(4, 'big') + val
    elif isinstance(typ, list):
        assert len(typ) == 1
        sub = b''.join([serialize(x, typ[0]) for x in val])
        return len(sub).to_bytes(4, 'big') + sub
    elif isinstance(typ, type):
        sub = b''.join([serialize(getattr(val, k), typ.fields[k]) for k in sorted(typ.fields.keys())])
        return len(sub).to_bytes(4, 'big') + sub
    raise Exception("Cannot serialize", val, typ)

def _deserialize(data, start, typ):
    if typ in ('hash32', 'address'):
        length = 20 if typ == 'address' else 32
        assert len(data) + start >= length
        return data[start: start+length], start+length
    elif isinstance(typ, str) and typ[:3] == 'int':
        length = int(typ[3:])
        assert length % 8 == 0
        assert len(data) + start >= length // 8
        return int.from_bytes(data[start: start+length//8], 'big'), start+length//8
    elif typ == 'bytes':
        length = int.from_bytes(data[start:start+4], 'big')
        assert len(data) + start >= 4+length
        return data[start+4: start+4+length], start+4+length
    elif isinstance(typ, list):
        assert len(typ) == 1
        length = int.from_bytes(data[start:start+4], 'big')
        pos, o = start + 4, []
        while pos < start + 4 + length:
            result, pos = _deserialize(data, pos, typ[0])
            o.append(result)
        assert pos == start + 4 + length
        return o, pos
    elif isinstance(typ, type):
        length = int.from_bytes(data[start:start+4], 'big')
        values = {}
        pos = start + 4
        for k in sorted(typ.fields.keys()):
            values[k], pos = _deserialize(data, pos, typ.fields[k])
        assert pos == start + 4 + length
        return typ(**values), pos
    raise Exception("Cannot deserialize", typ)

def deserialize(data, typ):
    return _deserialize(data, 0, typ)[0]

def eq(x, y):
    if hasattr(x, 'fields') and hasattr(y, 'fields'):
        for f in x.fields:
            if not eq(getattr(x, f), getattr(y, f)):
                print('Unequal:', x, y, f, getattr(x, f), getattr(y, f))
                return False
            return True
    else:
        return x == y

def deepcopy(x):
    if hasattr(x, 'fields'):
        vals = {}
        for f in x.fields.keys():
            vals[f] = deepcopy(getattr(x, f))
        return x.__class__(**vals)
    elif isinstance(x, list):
        return [deepcopy(y) for y in x]
    else:
        return x

def to_dict(x):
    if hasattr(x, 'fields'):
        vals = {}
        for f in x.fields.keys():
            vals[f] = to_dict(getattr(x, f))
        return vals
    elif isinstance(x, list):
        return [to_dict(y) for y in x]
    else:
        return x
