# Results: t095_smoke

Problem xorgauss (linear models: 0.50, best possible 0.955); success = best brain >= 0.95 TEST accuracy (picked on training accuracy). Brains: 2-8-2 tanh nets (42 weights+biases). Evolution: P=1000, cull 50% per generation. Budget: 2000 generations (= 2,000,000 network evaluations).

| method | seeds solved | network evaluations to success (median, range) | training samples processed (median) | test accuracy reached (median) |
|---|---|---|---|---|
| evolution (random noise + culling) | 1/1 | 34,000 (34,000-34,000) | 68,000,000 samples, forward only | 0.950 |
| random search (no culling) | 0/1 | nan (0-0) | nan samples, forward only | 0.931 |
| gradient descent (Adam) | 1/1 | - (1 network) | 39,200 samples, forward+backward | 0.950 |
