## Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

def compute_size(model):
    """
    Size of model (in MB).
    """

    res = 0
    for n, p in model.named_parameters():
        res += p.numel()

    return res * 4 / 1024 / 1024
