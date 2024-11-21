from .utils import int_to_big_endian, big_endian_to_int

class Serializable():
    @classmethod
    def _s(cls, obj):
        o = []
        for field, serializer in obj.__class__.fields:
            member = getattr(obj, field)
            if isinstance(serializer, Serializable):
                assert isinstance(member, serializer)
            o.append(serializer._s(member))
        return o

    @classmethod
    def _d(cls, data, *args, **kwargs):
        obj = cls(*args, **kwargs)
        assert len(data) == len(cls.fields)
        for datum, (field, serializer) in zip(data, cls.fields):
            setattr(obj, field, serializer._d(datum))
        return obj
        

def int_in_range(_min, _max):
    class c():
        @classmethod
        def _s(cls, x):
            assert isinstance(x, int) and x >= _min and x <= _max
            return int_to_big_endian(x)
        @classmethod
        def _d(cls, x):
            assert len(x) == 0 or x[0] != 0
            return big_endian_to_int(x)
    return c

big_endian_int = int_in_range(0, 2**256-1)
int256 = int_in_range(0, 2**2048-1)

def bytesn(n):
    class c():
        @classmethod
        def _s(cls, x):
            assert isinstance(x, bytes) and len(x) == n
            return x
        @classmethod
        def _d(cls, x):
            return x
    return c

hash32 = bytesn(32)

class binary():
    @classmethod
    def _s(cls, x):
        assert isinstance(x, bytes)
        return x
    @classmethod
    def _d(cls, x):
        return x

def CountableList(ser):
    class c():
        @classmethod
        def _s(cls, vals):
            assert isinstance(vals, list)
            return [ser._s(v) for v in vals]     
        @classmethod
        def _d(cls, vals):
            return [ser._d(v) for v in vals]
    return c
