# Results: t095

Problem xorgauss (linear models: 0.50, best possible 0.955); success = best brain >= 0.95 TEST accuracy (picked on training accuracy). Brains: 2-8-2 tanh nets (42 weights+biases). Evolution: P=1000, cull 50% per generation. Budget: 2000 generations (= 2,000,000 network evaluations).

| method | seeds solved | network evaluations to success (median, range) | training samples processed (median) | test accuracy reached (median) |
|---|---|---|---|---|
| evolution (random noise + culling) | 5/5 | 42,000 (26,000-65,000) | 84,000,000 samples, forward only | 0.952 |
| random search (no culling) | 0/5 | nan (0-0) | nan samples, forward only | 0.932 |
| gradient descent (Adam) | 5/5 | - (1 network) | 39,200 samples, forward+backward | 0.950 |
