from functools import wraps

def computeBefore(f):
    @wraps(f)
    def wrapper(self, *args, **kwargs):
        self.preCompute()
        return f(self, *args, *kwargs)
    return wrapper