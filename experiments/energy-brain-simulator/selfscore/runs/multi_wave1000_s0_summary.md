# Run wave1000_s0 at a glance

**Generation 2999 of 3000** | epoch 0 | 3.2 min | 0 total wipes so far

**Headline:** the best brain averages **59%** accuracy over all 24 problems (26% once chance is subtracted); solves **1** at 90%+.

## Epochs (a new epoch starts after every wipe)

| epoch | from gen | to gen | tests | start | score when it ended | ended by |
|---|---|---|---|---|---|---|
| 0 | 0 | running | 24 problems | all random | 0.562 | still running |

## Best brain, every problem (fixed reference test data; each # is 10% accuracy; chance-adjusted = how far above guessing)

| problem | accuracy | | chance-adjusted | in this epoch's tests |
|---|---|---|---|---|
| parity4 | 69% | `#######...` | +38% | yes |
| parity6 | 55% | `#####.....` | +9% | yes |
| majority5 | 100% | `##########` | +100% | yes |
| mux6 | 64% | `######....` | +28% | yes |
| adder2 | 44% | `####......` | +34% | yes |
| quadrants | 48% | `#####.....` | -3% | yes |
| moons | 76% | `########..` | +51% | yes |
| circles | 72% | `#######...` | +43% | yes |
| spirals | 49% | `#####.....` | -3% | yes |
| blobs3 | 44% | `####......` | +16% | yes |
| parity3of6 | 45% | `#####.....` | -9% | yes |
| cmp3 | 62% | `######....` | +25% | yes |
| addmod4 | 44% | `####......` | +25% | yes |
| and_or10 | 62% | `######....` | +25% | yes |
| majority7 | 82% | `########..` | +64% | yes |
| checker3 | 60% | `######....` | +21% | yes |
| sine | 43% | `####......` | -13% | yes |
| blobs4 | 38% | `####......` | +17% | yes |
| rings3 | 40% | `####......` | +11% | yes |
| xorgauss | 56% | `######....` | +13% | yes |
| moons_noisy15 | 68% | `#######...` | +37% | yes |
| xor2 | 75% | `########..` | +50% | yes |
| parity3 | 62% | `######....` | +25% | yes |
| xor2_d4 | 56% | `######....` | +12% | yes |

## What the random cull did (change in average accuracy, bucketed by that generation's cull draw)

| cull draw | generations | avg change in accuracy |
|---|---|---|
| 0%-20% | 28 | +0.0035 |
| 20%-40% | 48 | +0.0003 |
| 40%-60% | 45 | +0.0000 |
