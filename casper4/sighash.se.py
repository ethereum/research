# Fetches the char from calldata at position $x
macro calldatachar($x):
    div(calldataload($x), 2**248)

# Fetches the next $b bytes from calldata starting at position $x 
# Assumes that there is nothing important in memory at bytes 0..63
macro calldatabytes_as_int($x, $b):
    ~mstore(32-$b, calldataload($x))
    ~mload(0)

# Position in calldata
with pos = 0:
    # First char in calldata
    with c0 = calldatachar(0):
        # The start of the array must be in 192...255 because it represents
        # a list length
        # Length ++ body case
        if c0 < 248:
            pos = 1
        # Length of length ++ length ++ body case
        else:
            pos = (c0 - 246)
    # Start position of the list (save it)
    with startpos = pos:
        # Start position of the previous element
        with lastpos = 0:
            # Keep looping until we hit the end of the input
            while pos < ~calldatasize():
                # Next char in calldata
                with c = calldatachar(pos):
                    lastpos = pos
                    # Single byte 0x00...0x7f body case
                    if c < 128:
                        pos += 1
                    # Length ++ body case
                    elif c < 184:
                        pos += c - 127
                    # Length of length ++ length ++ body case
                    elif c < 192:
                        pos += calldatabytes_as_int(pos + 1, c - 183) + (c - 182)
         
            # Length of new RLP list
            with newlen = lastpos - startpos:
                # Length ++ body case
                if newlen < 56:
                    # Store length in the first byte
                    ~mstore8(0, 192 + newlen)
                    # Copy calldata right after length
                    ~calldatacopy(1, startpos, newlen)
                    # Return the hash
                    return(~sha3(0, 1 + newlen))
                else:
                    # The log256 of the length (ie. length of length)
                    # Can't go higher than 16777216 bytes due to gas limits
                    with _log = if(newlen < 256, 1, if(newlen < 65536, 2, 3)):
                        # Store the length
                        ~mstore(0, newlen)
                        # Store the length of the length right before the length
                        with 31minuslog = 31 - _log:
                            ~mstore8(31minuslog, 247 + _log)
                            # Store the rest of the data
                            ~calldatacopy(32, startpos, newlen)
                            # Return the hash
                            return(~sha3(31minuslog, 1 + _log + newlen))
