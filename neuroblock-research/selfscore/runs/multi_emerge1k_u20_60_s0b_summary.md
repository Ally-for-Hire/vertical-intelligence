# Run emerge1k_u20_60_s0b at a glance

**Generation 2999 of 3000** | epoch 0 | 5.5 min | 0 total wipes so far

**Headline:** the best brain averages **60%** accuracy over all 24 problems (27% once chance is subtracted); solves **0** at 90%+.

## Epochs (a new epoch starts after every wipe)

| epoch | from gen | to gen | tests | start | score when it ended | ended by |
|---|---|---|---|---|---|---|
| 0 | 0 | running | 24 problems | all random | 0.588 | still running |

## Best brain, every problem (fixed reference test data; each # is 10% accuracy; chance-adjusted = how far above guessing)

| problem | accuracy | | chance-adjusted | in this epoch's tests |
|---|---|---|---|---|
| parity4 | 56% | `######....` | +12% | yes |
| parity6 | 52% | `#####.....` | +3% | yes |
| majority5 | 81% | `########..` | +62% | yes |
| mux6 | 59% | `######....` | +19% | yes |
| adder2 | 38% | `####......` | +27% | yes |
| quadrants | 45% | `#####.....` | -9% | yes |
| moons | 88% | `#########.` | +77% | yes |
| circles | 66% | `#######...` | +33% | yes |
| spirals | 61% | `######....` | +21% | yes |
| blobs3 | 78% | `########..` | +68% | yes |
| parity3of6 | 56% | `######....` | +12% | yes |
| cmp3 | 72% | `#######...` | +44% | yes |
| addmod4 | 38% | `####......` | +17% | yes |
| and_or10 | 60% | `######....` | +20% | yes |
| majority7 | 69% | `#######...` | +38% | yes |
| checker3 | 54% | `#####.....` | +8% | yes |
| sine | 41% | `####......` | -19% | yes |
| blobs4 | 37% | `####......` | +16% | yes |
| rings3 | 39% | `####......` | +9% | yes |
| xorgauss | 56% | `######....` | +13% | yes |
| moons_noisy15 | 77% | `########..` | +55% | yes |
| xor2 | 75% | `########..` | +50% | yes |
| parity3 | 62% | `######....` | +25% | yes |
| xor2_d4 | 70% | `#######...` | +41% | yes |

## What the random cull did (change in average accuracy, bucketed by that generation's cull draw)

| cull draw | generations | avg change in accuracy |
|---|---|---|
| 20%-40% | 61 | +0.0036 |
| 40%-60% | 60 | -0.0014 |
