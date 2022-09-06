#!/usr/bin/env python

import numpy as np
import vquantizers as vq

if __name__ == '__main__':

        N = 128
        D = 128
        M = 128
        codebooks = 16

        X = np.random.randint(100, size=(N, D))
        Q = np.random.randint(100, size=(D, M))

        task = vq.MithralEncoder(codebooks)

        task.fit(X)

        X_enc = task.encode_X(X)

        luts, offset, scale = task.encode_Q(Q.T)

        W = task.dists_enc(X_enc, luts, False, offset, scale)

        print("W: ", W)
        print("-----")

        W_real = np.matmul(X, Q)
#       mse = np.square(W - W_real).mean()
#       print("mse: ", mse)
        print("offset: ", np.abs(W - W_real))