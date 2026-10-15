# r61 — the north-star result: store >> context window AND query only what is required.

r51 showed the store bridges ~M/128 windows (31x at M=4096). The read was DENSE (softmax over all M).
Here: read the big store with a top-k readout (k in 1,2,4,8, dense) at 7/15/31-window bridges.
If top-1 matches dense, the store holds K x the window AND each query touches only ONE of M slots.
Smoke (7 windows, M=1024): top-1 = top-8 = dense = 100%, touching 1/889 slots. r61 confirms at scale.
