import functools


def cuda_cast(func):

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        new_args = []
        for x in args:
            if hasattr(x, 'cuda'):
                x = x.cuda()
            new_args.append(x)
        new_kwargs = {}
        for k, v in kwargs.items():
            if hasattr(v, 'cuda'):
                v = v.cuda()
            elif isinstance(v, list) and hasattr(v[0], 'cuda'):
                v = [x.cuda() for x in v]
            new_kwargs[k] = v
        return func(*new_args, **new_kwargs)

    return wrapper


class AverageMeter(object):
    """Computes and stores the average and current value."""

    def __init__(self, apply_dist_reduce=False):
        self.apply_dist_reduce = apply_dist_reduce
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def get_val(self):
        return self.val

    def get_avg(self):
        return self.avg

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
