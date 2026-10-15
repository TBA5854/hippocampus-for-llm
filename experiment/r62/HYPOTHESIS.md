# r62 — harden the r61 north-star result

r61 skipped the matched control. Add: (1) SHUFFLE control (read a neighbour store -> chance),
(2) a 2nd model (r50_bpe16_s2), (3) a bigger scale (M=8192, 63 windows). If top-1 stays 100% real
and the shuffle control drops to chance, the sparse-read flagship is bulletproof.
