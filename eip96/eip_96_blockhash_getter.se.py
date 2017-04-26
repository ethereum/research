# Setting the block hash
if msg.sender == 2**160 - 2:
    with prev_block_number = block.number - 1:
        # Use storage fields 0..255 to store the last 256 hashes
        ~sstore(prev_block_number % 256, ~calldataload(0))
        # Use storage fields 256..511 to store the hashes of the last 256
        # blocks with block.number % 256 == 0
        if not (prev_block_number % 256):
            ~sstore(256 + (prev_block_number / 256) % 256, ~calldataload(0))
        # Use storage fields 512..767 to store the hashes of the last 256
        # blocks with block.number % 65536 == 0
        if not (prev_block_number % 65536):
            ~sstore(512 + (prev_block_number / 65536) % 256, ~calldataload(0))
# Getting the block hash
else:
    if ~calldataload(0) >= block.number:
        return(0)
    elif block.number - ~calldataload(0) <= 256:
        return(~sload(~calldataload(0) % 256))
    elif (not (~calldataload(0) % 256) and block.number - ~calldataload(0) <= 65536):
        return(~sload(256 + (~calldataload(0) / 256) % 256))
    elif (not (~calldataload(0) % 65536) and block.number - ~calldataload(0) <= 16777216):
        return(~sload(512 + (~calldataload(0) / 65536) % 256))
    else:
        return(0)
