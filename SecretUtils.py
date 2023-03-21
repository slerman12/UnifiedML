import torch


# MPS conversions  TODO Should I use this?
def convert(func):
    def _convert(*args, **kwargs):
        args = list(args)

        mps = False

        for i, item in enumerate(args):
            if isinstance(item, torch.Tensor) and item.device.name == 'mps':
                args[i] = item.to('cpu')
                mps = True

        for key, item in kwargs.items():
            if isinstance(item, torch.Tensor) and item.device.name == 'mps':
                kwargs[key] = item.to('cpu')
                mps = True

        return func(*args, **kwargs).to('mps') if mps else func(*args, **kwargs)
    return _convert


# torch.masked_select = convert(torch.masked_select)  # TODO Should I use this?
# torch.logsumexp = convert(torch.logsumexp)  # TODO Should I use this?


# torch.masked_select not supported on M1 Mac MPS by Pytorch
func = torch.masked_select
torch.masked_select = lambda input, mask, **out: \
    func(input.to('cpu'), mask.to('cpu'), **out).to('mps') if input.device.name == 'mps' else func(input, mask, **out)

# torch.masked_select not supported on M1 Mac MPS by Pytorch  TODO Just supress warning
# if args.device == 'mps':
#     func = torch.masked_select
#     torch.masked_select = lambda item, mask, **kwargs: func(item.to('cpu'), mask.to('cpu'), **kwargs).to('mps')
