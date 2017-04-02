with inp = ~calldataload(0):
    foo = inp
    exp = 0
    while foo >= 256:
        foo = ~div(foo, 256)
        exp += 1
    with x = ~div(inp, 16 ** exp):
        while 1:
            y = ~div(x + ~div(inp, x) + 1, 2)
            if x == y:
                return x
            x = y
