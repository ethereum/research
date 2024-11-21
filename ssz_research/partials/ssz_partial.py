from minimal_ssz import infer_type, is_basic, merkleize, pack_object, hash_tree_root, pack, is_power_of_two, ZERO_CHUNK, Vector, convert_to_list, is_top_level_dynamic, get_subtype_if_basic
from hash_function import hash

def last_power_of_two(x):
    return x if x <= 1 else 2 * last_power_of_two(x // 2)

def next_power_of_two(x):
    return x if x <= 1 else 2 * next_power_of_two((x + 1) // 2)

def concat_generalized_indices(x, y):
    return x * last_power_of_two(y) + y - last_power_of_two(y)

def rebase(objs, new_root):
    return {concat_generalized_indices(new_root, k): v for k,v in objs.items()}

def constrict_generalized_index(x, q):
    depth = last_power_of_two(x // q)
    o = depth + x - q * depth
    if concat_generalized_indices(q, o) != x:
        return None
    return o

def unrebase(objs, q):
    o = {}
    for k,v in objs.items():
        new_k = constrict_generalized_index(k, q)
        if new_k is not None:
            o[new_k] = v
    return o

def merkle_branch(chunks, index):
    tree = chunks[::]
    while not is_power_of_two(len(tree)):
        tree.append(ZERO_CHUNK)
    tree = [ZERO_CHUNK] * len(tree) + tree
    output = {}
    opos = len(tree) // 2 + index
    for i in range(len(tree) // 2 - 1, 0, -1):
        tree[i] = hash(tree[i * 2] + tree[i * 2 + 1])
        if i == opos // 2:
            output[opos ^ 1] = tree[opos ^ 1]
            opos //= 2
    return output

def get_size_of_basic_type(typ):
    if not isinstance(typ, str):
        raise Exception("Not basic")
    elif typ[:4] == 'uint' and typ[4:] in ['8', '16', '32', '64', '128', '256']:
        return int(typ[4:]) // 8
    elif typ == 'bool':
        return 1
    elif typ == 'byte':
        return 1
    else:
        raise Exception("Not basic")

def deserialize_basic(data, typ):
    if not isinstance(typ, str):
        raise Exception("Not basic")
    elif typ[:4] == 'uint' and typ[4:] in ['8', '16', '32', '64', '128', '256']:
        return int.from_bytes(data, 'little')
    elif typ == 'bool':
        return data == b'\x01'
    elif typ == 'byte':
        return data
    else:
        raise Exception("Not basic")

def filler(starting_position, chunk_count):
    at, skip, end = chunk_count, 1, next_power_of_two(chunk_count)
    value = ZERO_CHUNK
    o = {}
    while at < end:
        while at % (skip*2) == 0:
            skip *= 2
            value = hash(value + value)
        o[starting_position + at] = value
        at += skip
    return o

def ssz_all(value, typ=None, root=1):
    if typ is None:
        typ = infer_type(value)

    value_as_list = convert_to_list(value)
    chunks = pack_object(value, typ)    

    if is_top_level_dynamic(typ):
        starting_tree_index = root * 2 * next_power_of_two(len(chunks))
        output = {root*2+1: len(value_as_list).to_bytes(32, 'little')}
    else:
        starting_tree_index = root * next_power_of_two(len(chunks))
        output = {}

    if get_subtype_if_basic(typ):
        for i, chunk in enumerate(chunks):
            output[starting_tree_index+i] = chunk
        output = {**output, **filler(starting_tree_index, len(chunks))}
    else:
        for i, element in enumerate(value_as_list):
            output = {**output, **ssz_all(element, root=starting_tree_index + i)}
        output = {**output, **filler(starting_tree_index, len(value_as_list))}
    return output

def ssz_branch(value, path, typ=None):
    if typ is None:
        typ = infer_type(value)

    value_as_list = convert_to_list(value)
    chunks = pack_object(value, typ)    

    if is_basic(typ):
        if len(path) > 0:
            raise Exception("Empty path required for non-basic type: {}".format(typ))
        return {1: pack([value], typ)[0]}
    if len(path) == 0:
        return ssz_all(value)
    if isinstance(typ, list) and path[0] == '__len__':
        return {3: len(value).to_bytes(32, 'little'), 2: merkleize(chunks)}
    elif isinstance(typ, list) or (isinstance(typ, str) and typ[:5] == 'bytes'):
        assert isinstance(path[0], int), "Expected int: {}".format(path[0])
        index = path[0] * len(chunks) // len(value)
    elif hasattr(typ, 'fields'):
        assert isinstance(path[0], str), "Expected member variable: {}".format(path[0])
        index = list(typ.fields.keys()).index(path[0])
    else:
        raise Exception("Unrecognized type")
    first_section = merkle_branch(chunks, index)
    sub_position = next_power_of_two(len(chunks)) + index
    if (isinstance(typ, list) and len(typ) == 1) or typ == 'bytes':
        first_section = {3: len(value).to_bytes(32, 'little'), **rebase(first_section, 2)}
        sub_position = concat_generalized_indices(2, sub_position)
    if (isinstance(typ, list) and is_basic(typ[0])) or typ == 'bytes':
        second_section = {sub_position: chunks[index]}
    elif isinstance(typ, list):
        second_section = rebase(ssz_branch(value[path[0]], path[1:]), sub_position)
    else:
        second_section = rebase(ssz_branch(getattr(value, path[0]), path[1:]), sub_position)
    return {**first_section, **second_section}

def merge_ssz_branches(*branches):
    o = {}
    for branch in branches:
        # print(branch.keys())
        o = {**o, **branch}
    keys = sorted(o.keys())[::-1]
    pos = 0
    while pos < len(keys):
        k = keys[pos]
        if k in o and k^1 in o and k//2 not in o:
            o[k//2] = hash(o[k&-2] + o[k|1])
            keys.append(k // 2)
        pos += 1
    return {x: o[x] for x in o if not (x*2 in o and x*2+1 in o)}

def get_all_generalized_indices(obj, root=1):
    typ = infer_type(obj)
    if is_basic(typ):
        return [root]
    if (isinstance(typ, list) and len(typ) == 1) or typ == 'bytes':
        o, subroot = [root*2+1], root*2
    else:
        o, subroot = [], root
    if isinstance(typ, list) or (isinstance(typ, str) and typ[:5] == 'bytes'):
        if is_basic(typ[0]):
            item_size = get_size_of_basic_type(typ[0])
        elif isinstance(typ, str) and typ[:5] == 'bytes':
            item_size = 1
        else:
            item_size = 32
        length = (len(obj) - 1) * item_size // 32 + 1
        if item_size < 32:
            return o + [concat_generalized_indices(subroot, next_power_of_two(length) + i) for i in range(length)]
        return o + sum([get_all_generalized_indices(obj[i], concat_generalized_indices(subroot, next_power_of_two(length) + i)) for i in range(len(obj))], [])
    elif hasattr(typ, "fields"):
        return sum([get_all_generalized_indices(getattr(obj, field), concat_generalized_indices(root, next_power_of_two(len(typ.fields)) + i)) for i, field in enumerate(list(typ.fields.keys()))], [])
    else:
        raise Exception("Unknown type / path", typ, obj)
        
def get_generalized_indices(obj, path, root=1):
    typ = infer_type(obj)
    # print(path, root, typ)
    if len(path) == 0:
        o = get_all_generalized_indices(obj, root)
        return o
    elif ((isinstance(typ, list) and len(typ) == 1) or typ == 'bytes') and path[0] == '__len__':
        return [root * 2 + 1]
    elif ((isinstance(typ, list) and len(typ) == 2) or (isinstance(typ, str) and typ[:5] == 'bytes')) and path[0] == '__len__':
        return []
    elif isinstance(typ, list) or (isinstance(typ, str) and typ[:5] == 'bytes'):
        if is_basic(typ[0]):
            item_size = get_size_of_basic_type(typ[0])
        elif isinstance(typ, str) and typ[:5] == 'bytes':
            item_size = 1
        else:
            item_size = 32
        index = path[0] * item_size // 32
        length = (len(obj) - 1) * item_size // 32 + 1
        if len(typ) == 1 or typ == 'bytes':
            new_root = root * 2 * next_power_of_two(length) + index
            return [concat_generalized_indices(root, 3)] + get_generalized_indices(obj[path[0]], path[1:], new_root)
        else:
            new_root = root * next_power_of_two(length) + index
            return get_generalized_indices(obj[path[0]], path[1:], new_root)
    elif hasattr(typ, "fields"):
        new_root = root * next_power_of_two(len(typ.fields)) + list(typ.fields.keys()).index(path[0])
        return get_generalized_indices(getattr(obj, path[0]), path[1:], new_root)
    else:
        raise Exception("Unknown type / path", typ, obj)

def descend(obj, path):
    if len(path) == 0:
        return obj
    elif isinstance(path[0], int):
        return descend(obj[path[0]], path[1:])
    elif isinstance(path[0], str):
        return descend(getattr(obj, path[0]), path[1:])
    elif path[0] == '__len__':
        return len(obj)
    else:
        raise Exception("Unknown path object", path)

def get_proof_indices(tree_indices):
    # Get all indices touched by the proof
    keys = set(tree_indices)
    for i in tree_indices:
        x = i
        while x > 1:
            keys.add(x ^ 1)
            x //= 2
    keys = sorted(list(keys))[::-1]
    pos = 0
    while pos < len(keys):
        k = keys[pos]
        if k in keys and k^1 in keys and k//2 not in keys:
            keys.append(k // 2)
        pos += 1
    # Get indices that cannot be recalculated from earlier indices
    return [x for x in keys if not (x*2 in keys and x*2+1 in keys)]

class OutOfRangeException(Exception):
    pass

class SSZPartial():
    def __init__(self, typ, objects):
        self.objects = objects
        self.typ = typ
        if hasattr(self.typ, 'fields'):
            for field in self.typ.fields:
                try:
                    setattr(self, field, self.getter(field))
                except OutOfRangeException:
                    pass

    def getter(self, index):
        length = len(self)
        base_index = 2 if (isinstance(self.typ, list) and len(self.typ) == 1) or self.typ == 'bytes' else 1
        if isinstance(self.typ, list) and is_basic(self.typ[0]):
            item_size = get_size_of_basic_type(self.typ[0])
            chunk_index = index * item_size // 32
            chunk_count = (length * item_size + 32 - item_size) // 32
            tree_index = concat_generalized_indices(base_index, next_power_of_two(chunk_count) + chunk_index)
            if tree_index not in self.objects:
                raise OutOfRangeException("Do not have required data")
            return deserialize_basic(self.objects[tree_index][((item_size * index)%32):][:item_size], self.typ[0])
        elif isinstance(self.typ, list):
            tree_index = concat_generalized_indices(base_index, next_power_of_two(length) + index)
            return SSZPartial(self.typ[0], unrebase(self.objects, tree_index))
        elif hasattr(self.typ, 'fields'):
            chunk_index = list(self.typ.fields.keys()).index(index)
            tree_index = concat_generalized_indices(base_index, next_power_of_two(length) + chunk_index)
            subtype = self.typ.fields[index]
            if is_basic(subtype):
                if tree_index not in self.objects:
                    raise OutOfRangeException("Do not have required data")
                return deserialize_basic(self.objects[tree_index][:get_size_of_basic_type(subtype)], subtype)
            else:
                return SSZPartial(self.typ.fields[index], unrebase(self.objects, tree_index))
        elif self.typ[:5] == 'bytes':
            chunk_index = index // 32
            chunk_count = (length + 31) // 32
            tree_index = concat_generalized_indices(base_index, next_power_of_two(chunk_count) + chunk_index)
            if tree_index not in self.objects:
                raise OutOfRangeException("Do not have required data")
            return self.objects[tree_index][index%32]
        else:
            raise Exception("Unsupported type: {}".format(self.typ))

    def __getitem__(self, index):
        return self.getter(index)

    def __len__(self):
        if (isinstance(self.typ, list) and len(self.typ) == 1) or self.typ == 'bytes':
            return int.from_bytes(self.objects[3], 'little')
        elif isinstance(self.typ, list) and len(self.typ) == 2:
            return self.typ[1]
        elif isinstance(self.typ, str) and self.typ[:5] == 'bytes':
            return int(self.typ[5:])
        elif hasattr(self.typ, 'fields'):
            return len(self.typ.fields)
        else:
            raise Exception("Unsupported type: {}".format(self.typ))

    def full_value(self):
        if self.typ == 'bytes':
            return bytes([self.getter(i) for i in range(len(self))])
        elif isinstance(self.typ, list) and len(self.typ) == 1:
            return [self[i] for i in range(len(self))]
        elif isinstance(self.typ, list) and len(self.typ) == 2:
            return Vector([self[i] for i in range(len(self))])
        elif hasattr(self.typ, "fields"):
            full_value = lambda x: x.full_value() if hasattr(x, 'full_value') else x
            return self.typ(**{field: full_value(self.getter(field)) for field in self.typ.fields})

    def root(self):
        o = {**self.objects}
        keys = sorted(o.keys())[::-1]
        pos = 0
        while pos < len(keys):
            k = keys[pos]
            if k in o and k^1 in o and k//2 not in o:
                o[k//2] = hash(o[k&-2] + o[k|1])
                keys.append(k // 2)
            pos += 1
        return o[1]

    def __str__(self):
        return str(self.full_value())
