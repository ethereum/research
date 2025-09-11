from zorch import koalabear

def hash(*args):
    o = koalabear.KoalaBear(koalabear.koalabear_field.arange(8))
    for i, arg in enumerate(args):
        o += koalabear.KoalaBear.sum(o, axis=0)
        if arg.ndim == 0 and isinstance(arg, koalabear.KoalaBear):
            o[1] += arg
            o = (o + i) ** 5
        elif isinstance(arg, koalabear.KoalaBear):
            for j in range(0, arg.shape[0], 8):
                width = min(8, arg.shape[0]-j)
                o[:width] += arg[j:j+width]
                o = (o + i) ** 5
        elif arg.ndim == 0 and isinstance(arg, koalabear.ExtendedKoalaBear):
            o[:4] += koalabear.KoalaBear(arg.value)
            o = (o + i) ** 5
        else:
            raise Exception("not implemented")
    o += koalabear.KoalaBear.sum(o, axis=0)
    return o
