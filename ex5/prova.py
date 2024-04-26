import numpy as np
import pandas as pd

SEQ_ITEM_TYPE = np.dtype([("d", "int32"), ("l", "U50"), ("v", "float32")])


elemseq = np.array(
    [(1, "a", 1.0), (2, "b", 2.0), (3, "c", 3.0), (4, "d", 4.0), (5, "e", 5.0)],
    dtype=SEQ_ITEM_TYPE,
)

elemseq2 = np.array(
    [(6, "f", 6.0), (7, "g", 7.0), (9, "i", 9.0), (10, "j", 10.0)],
    dtype=SEQ_ITEM_TYPE,
)


dsseq = np.empty((2), dtype=object)
dsseq[0] = elemseq
dsseq[1] = elemseq2

print("dsseq", dsseq)
print("dsseq[0]", dsseq[0])
print("dsseq[0][0]", dsseq[0][0])

print("type(dsseq)", type(dsseq))
print("type(dsseq[0])", type(dsseq[0]))
print("type(dsseq[0][0])", type(dsseq[0][0]))

print("dsseq.shape", dsseq.shape)
print("dsseq[0].shape", dsseq[0].shape)
print("dsseq[0][0].shape", dsseq[0][0].shape)

print("dsseq", dsseq)
print("type(dsseq)", type(dsseq))

print("dsseq[0,:]", dsseq[0, :])
print("dsseq[0,:].shape", dsseq[0, :].shape)
