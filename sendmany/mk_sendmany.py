from ethereum import utils

def mk_multisend_code(payments): # expects a dictionary, {address: wei}
    kode = b''
    for address, wei in payments.items():
        kode += b'\x60\x00\x60\x00\x60\x00\x60\x00' # 0 0 0 0
        encoded_wei = utils.encode_int(wei) or b'\x00'
        kode += utils.ascii_chr(0x5f + len(encoded_wei)) + encoded_wei # value
        kode += b'\x73' + utils.normalize_address(address) # to 
        kode += b'\x60\x00\xf1\x50' # 0 CALL POP
    kode += b'\x33\xff' # CALLER SELFDESTRUCT
    return kode

def get_multisend_gas(payments):
    o = 26002 # 21000 + 2 (CALLER) + 5000 (SELFDESTRUCT)
    for address, wei in payments.items():
        encoded_wei = utils.encode_int(wei) or b'\x00'
        # 20 bytes in txdata for address = 1360
        # bytes in txdata for wei = 68 * n
        # gas for pushes and pops = 3 * 7 + 2 = 23
        # CALL = 9700 + 25000 (possible if new account)
        o += 1360 + 68 * len(encoded_wei) + 23 + 34700
    return o
