# Run peak10_s0 at a glance

**Generation 2999 of 3000** | epoch 0 | 3.6 min | 0 total wipes so far

**Headline:** the best brain averages **57%** accuracy over all 24 problems (22% once chance is subtracted); solves **1** at 90%+.

## Epochs (a new epoch starts after every wipe)

| epoch | from gen | to gen | tests | start | score when it ended | ended by |
|---|---|---|---|---|---|---|
| 0 | 0 | running | 24 problems | all random | 0.537 | still running |

## Best brain, every problem (fixed reference test data; each # is 10% accuracy; chance-adjusted = how far above guessing)

| problem | accuracy | | chance-adjusted | in this epoch's tests |
|---|---|---|---|---|
| parity4 | 50% | `#####.....` | +0% | yes |
| parity6 | 53% | `#####.....` | +6% | yes |
| majority5 | 91% | `#########.` | +81% | yes |
| mux6 | 59% | `######....` | +19% | yes |
| adder2 | 38% | `####......` | +27% | yes |
| quadrants | 48% | `#####.....` | -5% | yes |
| moons | 86% | `#########.` | +71% | yes |
| circles | 74% | `#######...` | +49% | yes |
| spirals | 52% | `#####.....` | +3% | yes |
| blobs3 | 56% | `######....` | +34% | yes |
| parity3of6 | 50% | `#####.....` | +0% | yes |
| cmp3 | 64% | `######....` | +28% | yes |
| addmod4 | 25% | `##........` | +0% | yes |
| and_or10 | 62% | `######....` | +24% | yes |
| majority7 | 62% | `######....` | +25% | yes |
| checker3 | 54% | `#####.....` | +9% | yes |
| sine | 46% | `#####.....` | -8% | yes |
| blobs4 | 30% | `###.......` | +6% | yes |
| rings3 | 40% | `####......` | +11% | yes |
| xorgauss | 54% | `#####.....` | +9% | yes |
| moons_noisy15 | 61% | `######....` | +22% | yes |
| xor2 | 75% | `########..` | +50% | yes |
| parity3 | 62% | `######....` | +25% | yes |
| xor2_d4 | 69% | `#######...` | +38% | yes |

## What the random cull did (change in average accuracy, bucketed by that generation's cull draw)

| cull draw | generations | avg change in accuracy |
|---|---|---|
| 0%-20% | 121 | +0.0009 |
