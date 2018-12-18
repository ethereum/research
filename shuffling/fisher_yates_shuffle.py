from hashlib import blake2s

def hash(x): return blake2s(x).digest()[:32]

def fisher_yates_shuffle(values,  seed):
    """
    Returns the shuffled ``values`` with ``seed`` as entropy.
    """
    values_count = len(values)

    # Entropy is consumed from the seed in 3-byte (24 bit) chunks.
    rand_bytes = 3
    # The highest possible result of the RNG.
    rand_max = 2 ** (rand_bytes * 8) - 1

    # The range of the RNG places an upper-bound on the size of the list that
    # may be shuffled. It is a logic error to supply an oversized list.
    assert values_count < rand_max

    output = [x for x in values]
    source = seed
    index = 0
    while index < values_count - 1:
        # Re-hash the `source` to obtain a new pattern of bytes.
        source = hash(source)
        # Iterate through the `source` bytes in 3-byte chunks.
        for position in range(0, 32 - (32 % rand_bytes), rand_bytes):
            # Determine the number of indices remaining in `values` and exit
            # once the last index is reached.
            remaining = values_count - index
            if remaining == 1:
                break

            # Read 3-bytes of `source` as a 24-bit big-endian integer.
            sample_from_source = int.from_bytes(source[position:position + rand_bytes], 'big')

            # Sample values greater than or equal to `sample_max` will cause
            # modulo bias when mapped into the `remaining` range.
            sample_max = rand_max - rand_max % remaining

            # Perform a swap if the consumed entropy will not cause modulo bias.
            if sample_from_source < sample_max:
                # Select a replacement index for the current index.
                replacement_position = (sample_from_source % remaining) + index
                # Swap the current index with the replacement index.
                output[index], output[replacement_position] = output[replacement_position], output[index]
                index += 1
            else:
                # The sample causes modulo bias. A new sample should be read.
                pass

    return output
