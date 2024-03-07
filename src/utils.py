

def throw(ex):
    raise ex

def curried_throw(ex):
    return lambda *_, **__: throw(ex)
