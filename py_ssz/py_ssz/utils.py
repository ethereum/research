from math import ceil

def int_to_big_endian(value):
    byte_length = max(ceil(value.bit_length() / 8), 1)
    return (value).to_bytes(byte_length, byteorder='big')


def big_endian_to_int(value):
    return int.from_bytes(value, byteorder='big')
