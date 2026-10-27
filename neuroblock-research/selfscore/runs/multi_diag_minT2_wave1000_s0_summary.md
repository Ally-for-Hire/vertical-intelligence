# Run diag_minT2_wave1000_s0 at a glance

**Generation 2900 of 3000** | epoch 0 | 9.9 min | 0 total wipes so far

**Headline:** the best brain averages **63%** accuracy over all 24 problems (32% once chance is subtracted); solves **2** at 90%+.

## Epochs (a new epoch starts after every wipe)

| epoch | from gen | to gen | tests | start | score when it ended | ended by |
|---|---|---|---|---|---|---|
| 0 | 0 | running | 24 problems | all random | 0.614 | still running |

## Best brain, every problem (fixed reference test data; each # is 10% accuracy; chance-adjusted = how far above guessing)

| problem | accuracy | | chance-adjusted | in this epoch's tests |
|---|---|---|---|---|
| parity4 | 69% | `#######...` | +38% | yes |
| parity6 | 56% | `######....` | +12% | yes |
| majority5 | 100% | `##########` | +100% | yes |
| mux6 | 66% | `#######...` | +31% | yes |
| adder2 | 50% | `#####.....` | +42% | yes |
| quadrants | 63% | `######....` | +25% | yes |
| moons | 87% | `#########.` | +73% | yes |
| circles | 75% | `#######...` | +49% | yes |
| spirals | 57% | `######....` | +13% | yes |
| blobs3 | 50% | `#####.....` | +25% | yes |
| parity3of6 | 47% | `#####.....` | -6% | yes |
| cmp3 | 64% | `######....` | +28% | yes |
| addmod4 | 31% | `###.......` | +8% | yes |
| and_or10 | 64% | `######....` | +29% | yes |
| majority7 | 95% | `#########.` | +89% | yes |
| checker3 | 62% | `######....` | +23% | yes |
| sine | 47% | `#####.....` | -6% | yes |
| blobs4 | 54% | `#####.....` | +39% | yes |
| rings3 | 41% | `####......` | +11% | yes |
| xorgauss | 56% | `######....` | +13% | yes |
| moons_noisy15 | 76% | `########..` | +52% | yes |
| xor2 | 75% | `########..` | +50% | yes |
| parity3 | 62% | `######....` | +25% | yes |
| xor2_d4 | 56% | `######....` | +12% | yes |

## What the random cull did (change in average accuracy, bucketed by that generation's cull draw)

| cull draw | generations | avg change in accuracy |
|---|---|---|
| 0%-20% | 24 | +0.0031 |
| 20%-40% | 48 | +0.0014 |
| 40%-60% | 45 | +0.0005 |
