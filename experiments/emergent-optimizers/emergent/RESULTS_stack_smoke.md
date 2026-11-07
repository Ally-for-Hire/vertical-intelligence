# Results: stack_smoke

Problem xorgauss (linear models: 0.50, best possible 0.955); success = best brain >= 0.95 TEST accuracy (picked on training accuracy). Brains: 2-8-2 tanh nets (42 weights+biases). Evolution: P=100, cull 50% per generation. Budget: 4000 generations (= 400,000 network evaluations).

| method | seeds solved | network evaluations to success (median, range) | training samples processed (median) | test accuracy reached (median) |
|---|---|---|---|---|
| undo + barred types + separate repair stack + culling | 1/1 | 68,907 (68,907-68,907) | 137,814,000 samples, forward only | 0.953 |
