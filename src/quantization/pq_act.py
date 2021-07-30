# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import torch

from .em_act import EM
from utils.reshape import reshape_weight, reshape_back_weight, reshape_activations, reshape_activations_pq


class PQ_act(EM):
    """
    Quantizes the layer M by taking into account the input activations.
    The columns are split into n_blocks blocks and a *joint* quantizer
    is learnt for all the blocks.

    Args:
        - in_features: future size(0) of the weight matrix
        - n_centroids: number of centroids per subquantizer
        - n_iter: number of k-means iterations
        - n_blocks: number of subquantizers

    Remarks:
        - For efficiency, we subsample the input activations
    """

    def __init__(self, M, eps=1e-8,
                 n_blocks=8, n_centroids=512, n_iter=20):# k=3, stride=(1, 1), padding=(1, 1), groups=1):
        super(PQ_act, self).__init__(n_centroids, M, eps=eps)
        self.n_blocks = n_blocks
        self.n_iter = n_iter
        self._reshape(M)
        # sanity check
        assert self.M.size(1) % n_blocks == 0, "n_blocks must be a multiple of channels"
        # initialize centroids
        M_reshaped = self.sample_weights()
        self.initialize_centroids(M_reshaped)

    def _reshape(self, M):
        """
        Rehshapes if conv or fully-connected.
        """

        self.M = reshape_activations_pq(M)

    def unroll_activations(self, in_activations):
        """
        Unroll activations.
        """

        return torch.cat(in_activations.chunk(self.n_blocks, dim=1), dim=0)

    def unroll_weight(self, M):
        """
        Unroll weights.
        """

        return torch.cat(M.chunk(self.n_blocks, dim=1), dim=2)

    def sample_activations(self):
        """
        Sample activations.
        """

        # get indices
        in_features = self.M.size(1)
        indices = torch.randint(low=0, high=self.in_activations.size(0), size=(self.n_samples // in_features,)).long()

        # sample current in_activations
        in_activations = self.unroll_activations(self.in_activations[indices])
        return in_activations.cuda()

    def sample_weights(self):
        """
        Sample weights (no sampling done here, only the unrolling).
        """

        return self.unroll_weight(self.M).cuda()

    def encode(self):
        """
        Args:
            - in_activations: input activations of size (n_samples x in_features)
            - M: weight matrix of the layer, of size (in_features x out_features)
        """

        # initialize sampling
        M_reshaped = self.sample_weights()

        # perform EM training steps
        for i in range(self.n_iter):
            self.centroids = self.step(M_reshaped, i)

    def decode(self, redo=False):
        """
        Args:
            - in_activations: input activations of size (n_samples x in_features)d
            - M: weight matrix of the layer, of size (in_features x out_features)
        """
        # use given activations to assign weightsgiven self.centroids
        if redo:
            M_reshaped = self.sample_weights()
            assignments = self.assign(M_reshaped)
            self.assignments = assignments
        else:
            assignments = self.assignments

        M_hat_reshaped = torch.cat(self.centroids[assignments].t().chunk(self.n_blocks, dim=1), dim=0)
        return reshape_back_weight(M_hat_reshaped, k=self.k, conv=self.conv)
