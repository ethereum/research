from foo import List, Vector, Bytes, BytesN, ValueCheckError

def assert_pass(typ, value):
    x = typ(*value) if issubclass(typ, List) or issubclass(typ, Vector) else typ(value)
    for i in range(len(value)):
        assert x[i] == value[i]

def assert_invalid(typ, value):
    try:
        x = typ(*value) if issubclass(typ, List) or issubclass(typ, Vector) else typ(value)
        passed = True
    except ValueCheckError:
        passed = False
    assert not passed

assert isinstance(List[int, 3](10, 20, 30), List[int, 3])
assert not isinstance(List[int, 3](10, 20, 30), List[int, 5])
assert not isinstance(List[int, 3](10, 20, 30), List[bool, 3])
assert isinstance(List[int, 3](10, 20, 30), List)
assert isinstance(Vector[int, 3](10, 20, 30), Vector[int, 3])
assert not isinstance(Vector[int, 3](10, 20, 30), List[int, 3])
assert isinstance(Bytes[3](b"cow"), Bytes[3])
assert not isinstance(Bytes[3](b"cow"), Bytes[5])
assert isinstance(Bytes[3](b"cow"), Bytes)
assert_pass(List[int, 3], [])
assert_pass(List[int, 3], [10])
assert_pass(List[int, 3], [10, 20])
assert_pass(List[int, 3], [10, 20, 30])
assert_invalid(List[int, 3], [10, 20, 30, 40])
assert_invalid(List[int, 3], ["dog", 20, 30])
assert_pass(Vector[int, 3], [10, 20, 30])
assert_invalid(Vector[int, 3], [10])
assert_invalid(Vector[int, 3], [10, 20, 30, 40])
assert_invalid(Vector[int, 3], [10, "cow", 30])
assert_pass(Vector[Vector[int, 2], 2], [Vector[int, 2](10, 20), Vector[int, 2](30, 40)])
assert_invalid(Vector[Vector[int, 2], 2], [Vector[int, 2](10, 20), Vector[int, 3](30, 40, 50)])
assert_invalid(Vector[Vector[int, 2], 2], [Vector[int, 2](10, 20), Vector[int, 2](30, 40), Vector[int, 2](50, 60)])
assert_invalid(Vector[Vector[int, 2], 2], [Vector[int, 2](10, 20), List[int, 2](30, 40)])
assert_pass(Bytes[3], b'ow')
assert_pass(Bytes[3], b'cow')
assert_invalid(Bytes[3], b'crow')
assert_invalid(Bytes[3], [10, 20, 30])
assert_invalid(BytesN[3], b'ow')
assert_pass(BytesN[3], b'cow')
assert_invalid(BytesN[3], b'crow')
assert_invalid(BytesN[3], [10, 20, 30])
x = List[int, 3](10, 20, 30)
x[2] = 100
assert x == List[int, 3](10, 20, 100)
print("Tests passed")
