def get_default_value(typ):
    if typ == int:
        return 0
    else:
        return typ.default()

def type_check(typ, value):
    if typ == int:
        return isinstance(value, int)
    else:
        return typ.value_check(value)

class AbstractListMeta(type):
    def __new__(cls, class_name, parents, attrs):
        out = type.__new__(cls, class_name, parents, attrs)
        if 'elem_type' in attrs and 'length' in attrs:
            setattr(out, 'elem_type', attrs['elem_type'])
            setattr(out, 'length', attrs['length'])
        return out

    child = lambda self: AbstractList

    def __getitem__(self, params):
        if not isinstance(params, tuple) or len(params) != 2:
            raise Exception("List must be instantiated with two args: elem type and length")
        o = self.__class__(self.__name__, (self.child(),), {'elem_type': params[0], 'length': params[1]})
        o._name = 'AbstractList'
        return o

    def __instancecheck__(self, obj):
        if not issubclass(obj.__class__, self.child()):
            return False
        if hasattr(self, 'elem_type'):
            return obj.__class__.elem_type == self.elem_type and obj.__class__.length == self.length
        return True

class ValueCheckError(Exception):
    pass

class AbstractList(metaclass=AbstractListMeta):
    def __init__(self, *args):
        items = args[0] if len(args) else self.default()
            
        if not self.value_check(items):
            raise ValueCheckError("Bad input for class {}: {}".format(self.__class__, items))
        self.items = items
    
    def value_check(self, value):
        for v in value:
            if not isinstance(v, self.__class__.elem_type):
                return False
        return True

    def default(self):
        raise Exception("Not implemented")

    def __getitem__(self, i):
        return self.items[i]

    def __len__(self):
        return len(self.items)

    def __repr__(self):
        return repr(self.items)

class ListMeta(AbstractListMeta):
    child = lambda self: List

class List(AbstractList, metaclass=ListMeta):
    def value_check(self, value):
        return len(value) <= self.__class__.length and super().value_check(value)

    def default(self):
        return []

class VectorMeta(AbstractListMeta):
    child = lambda self: Vector

class Vector(AbstractList, metaclass=VectorMeta):
    def value_check(self, value):
        return len(value) == self.__class__.length and super().value_check(value)

    def default(self):
        return [get_default_value(self.__class__.elem_type) for _ in range(self.__class__.length)]

class BytesMeta(AbstractListMeta):
    child = lambda self: Bytes

    def __getitem__(self, params):
        if not isinstance(params, int):
            raise Exception("Bytes must be instantiated with one arg: length")
        o = self.__class__(self.__name__, (self.child(),), {'length': params})
        o._name = 'Bytes'
        return o

class Bytes(AbstractList, metaclass=BytesMeta):
    def value_check(self, value):
        return len(value) <= self.__class__.length and isinstance(value, bytes)

    def default(self):
        return b''

class BytesNMeta(AbstractListMeta):
    child = lambda self: BytesN

    def __getitem__(self, params):
        if not isinstance(params, int):
            raise Exception("Bytes must be instantiated with one arg: length")
        o = self.__class__(self.__name__, (self.child(),), {'length': params})
        o._name = 'Bytes'
        return o

class BytesN(AbstractList, metaclass=BytesNMeta):
    def value_check(self, value):
        return len(value) == self.__class__.length and isinstance(value, bytes)

    def default(self):
        return b'\x00' * self.__class__.length
