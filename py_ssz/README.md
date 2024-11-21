## Simple serialization - an alternative to RLP that's much simpler.

### The spec:

    encode(string) = enc3b(len(string)) + string

Where `enc3b(x)` encodes `x` as three bytes, in big-endian format

    encode(list) = enc3b(8388608 + sum(len([encode(x) for x in list]))) + \
                   b''.join([encode(x) for x in list])

And that's it!

Examples:

    cow -> \x00\x00\x03cow

    [dog, horse] -> \x80\x00\x0e\x00\x00\x03dog\x00\x00\x05horse
