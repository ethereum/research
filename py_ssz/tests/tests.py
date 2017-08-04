from py_ssz import encode, decode
from py_ssz.serializers import big_endian_int, binary, hash32, CountableList, Serializable
from py_ssz.utils import int_to_big_endian

assert decode(encode(b'cow')) == b'cow'
assert decode(encode(123)) == int_to_big_endian(123)
assert decode(encode([b'cow', b'horse', b'pig'])) == [b'cow', b'horse', b'pig']
assert decode(encode([b'cow', b'horse', [b'o', b'q']])) == [b'cow', b'horse', [b'o', b'q']]
assert decode(encode([[], [[]], [[], [[]]]])) == [[], [[]], [[], [[]]]]

class Foo(Serializable):
    fields = [
        ('bar', big_endian_int),
        ('baz', binary)
    ]
    def __init__(self, bar=0, baz=b''):
        self.bar = bar
        self.baz = baz
    
assert encode(Foo(3, b'cow')) == encode([3, b'cow'])
foo2 = decode(encode(Foo(3, b'cow')), Foo)
assert foo2.bar == 3 and foo2.baz == b'cow'

class Foo2(Serializable):
    fields = [
        ('bat', Foo),
        ('bau', CountableList(big_endian_int)),
        ('bav', CountableList(Foo))
    ]

x = Foo2()
x.bat = Foo(3, b'cow')
x.bau = [4,5,6,7,8]
x.bav = [Foo(5, b'horse'), Foo(7, b'mongoose')]
assert encode(x) == encode([[3, b'cow'], [4,5,6,7,8], [[5, b'horse'], [7, b'mongoose']]])
