# r64 — CSA2 candidate-pool read (shared store + reuse)

Make the O(1)-slot read (r61/r62) also O(1)-COMPUTE: restrict scoring to an index-selected pool.
token = exact token-hash (our layer-0 key), ann = LSH bucket (fuzzy keys). box2 ablates none/token/ann
at 15/31/63-window stores; box1 tests cross-layer reuse (pool built once at anchor vs per read layer).
Smoke: token 1/889, ann 89/889, both 100% = dense. Target: recall = dense with candidates <<< M, at scale.
