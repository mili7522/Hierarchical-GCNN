import numpy as np
import sklearn
import sklearn.metrics.pairwise

V = np.random.rand(8, 6)

# Euclidean distance
D = sklearn.metrics.pairwise.euclidean_distances(V, V)

d1 = (V**2).sum(1, keepdims = True)
d2 = np.dot(V,V.T)
D1 = np.sqrt(d1 - 2 * d2 + d1.T)

# Mahalanobis distance
M = np.diag(np.random.rand(V.shape[1]))

d1 = (V * np.dot(V, M)).sum(1, keepdims = True)
d2 = np.dot(V, np.dot(M, V.T))
D1 = np.sqrt(d1 - 2 * d2 + d1.T)

# Check manually
D = np.zeros((len(V), len(V)))
for i in range(len(V)):
    for j in range(len(V)):
        diff = (V[i] - V[j]).reshape((1,-1))
        D[i,j] = np.sqrt(np.dot(np.dot(diff, M), diff.T))


### Multidimensional:
# Matching the dimensions of the Graph-CNN
# V = np.random.rand(1, 8, 6)

# M = np.diag(np.random.rand(V.shape[-1]))

# d1 = (V * np.dot(V, M)).sum(-1, keepdims = True)  # This is the same compared to if V was squeezed
# d2 = np.dot(V, np.tensordot(M, V.T.reshape((1, V.shape[-1], -1))) )
# D1 = np.sqrt(d1 - 2 * d2 + d1.T)

# D = np.zeros((len(A), len(A)))
# for i in range(len(A)):
#     for j in range(len(A)):
#         diff = (A[i] - A[j]).reshape((1,-1))
#         D[i,j] = np.sqrt(np.dot(np.dot(diff, M), diff.T))