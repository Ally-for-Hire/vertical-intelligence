# Results: smoke

Problem xorgauss (linear models: 0.50, best possible 0.955); success = best brain >= 0.9 TEST accuracy (picked on training accuracy). Brains: 2-8-2 tanh nets (42 weights+biases). Evolution: P=1000, cull 50% per generation. Budget: 2000 generations (= 2,000,000 network evaluations).

| method | seeds solved | network evaluations to success (median, range) | training samples processed (median) | test accuracy reached (median) |
|---|---|---|---|---|
| evolution (random noise + culling) | 1/1 | 8,000 (8,000-8,000) | 16,000,000 samples, forward only | 0.900 |
| random search (no culling) | 1/1 | 5,000 (5,000-5,000) | 10,000,000 samples, forward only | 0.900 |
| gradient descent (Adam) | 1/1 | - (1 network) | 19,200 samples, forward+backward | 0.907 |
