# Results: undo_vs_noise_P1000_c0.9

Problem xorgauss (linear models: 0.50, best possible 0.955); success = best brain >= 0.95 TEST accuracy (picked on training accuracy). Brains: 2-8-2 tanh nets (42 weights+biases). Evolution: P=1000, cull 90% per generation. Budget: 4000 generations (= 4,000,000 network evaluations).

| method | seeds solved | network evaluations to success (median, range) | training samples processed (median) | test accuracy reached (median) |
|---|---|---|---|---|
| evolution (random noise + culling) | 5/5 | 10,000 (7,000-14,000) | 20,000,000 samples, forward only | 0.952 |
| evolution with undo + barred change types + culling | 2/5 | 13,500 (10,000-17,000) | 27,000,000 samples, forward only | 0.943 |
