# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

def dynamic_sampling(layer):
    """
    Number of activations (after reshaping) to sample from a given layer (see )
    """

    if 'layer1' in layer or 'layer2' in layer:
        return 1000
    elif 'layer3' in layer or 'layer4' in layer:
        return 25000
    elif 'fc' in layer or 'classifier' in layer:
        return 5000
    else:
        return ValueError(layer)
