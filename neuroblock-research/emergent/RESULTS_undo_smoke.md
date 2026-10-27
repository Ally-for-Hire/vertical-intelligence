# Results: undo_smoke

Problem xorgauss (linear models: 0.50, best possible 0.955); success = best brain >= 0.95 TEST accuracy (picked on training accuracy). Brains: 2-8-2 tanh nets (42 weights+biases). Evolution: P=100, cull 50% per generation. Budget: 4000 generations (= 400,000 network evaluations).

| method | seeds solved | network evaluations to success (median, range) | training samples processed (median) | test accuracy reached (median) |
|---|---|---|---|---|
| evolution with undo + barred change types + culling | 1/1 | 9,100 (9,100-9,100) | 18,200,000 samples, forward only | 0.953 |
