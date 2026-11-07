# Run emerge1k_wave_blend_s0 at a glance

**Generation 2999 of 3000** | epoch 0 | 3.1 min | 0 total wipes so far

**Headline:** the best brain averages **59%** accuracy over all 24 problems (25% once chance is subtracted); solves **1** at 90%+.

## Epochs (a new epoch starts after every wipe)

| epoch | from gen | to gen | tests | start | score when it ended | ended by |
|---|---|---|---|---|---|---|
| 0 | 0 | running | 24 problems | all random | 0.564 | still running |

## Best brain, every problem (fixed reference test data; each # is 10% accuracy; chance-adjusted = how far above guessing)

| problem | accuracy | | chance-adjusted | in this epoch's tests |
|---|---|---|---|---|
| parity4 | 62% | `######....` | +25% | yes |
| parity6 | 55% | `#####.....` | +9% | yes |
| majority5 | 94% | `#########.` | +88% | yes |
| mux6 | 59% | `######....` | +19% | yes |
| adder2 | 38% | `####......` | +27% | yes |
| quadrants | 49% | `#####.....` | -1% | yes |
| moons | 86% | `#########.` | +73% | yes |
| circles | 71% | `#######...` | +41% | yes |
| spirals | 51% | `#####.....` | +1% | yes |
| blobs3 | 83% | `########..` | +75% | yes |
| parity3of6 | 48% | `#####.....` | -3% | yes |
| cmp3 | 62% | `######....` | +25% | yes |
| addmod4 | 25% | `##........` | +0% | yes |
| and_or10 | 62% | `######....` | +23% | yes |
| majority7 | 69% | `#######...` | +38% | yes |
| checker3 | 53% | `#####.....` | +7% | yes |
| sine | 45% | `#####.....` | -10% | yes |
| blobs4 | 41% | `####......` | +21% | yes |
| rings3 | 40% | `####......` | +10% | yes |
| xorgauss | 54% | `#####.....` | +9% | yes |
| moons_noisy15 | 63% | `######....` | +26% | yes |
| xor2 | 75% | `########..` | +50% | yes |
| parity3 | 62% | `######....` | +25% | yes |
| xor2_d4 | 67% | `#######...` | +34% | yes |

## What the random cull did (change in average accuracy, bucketed by that generation's cull draw)

| cull draw | generations | avg change in accuracy |
|---|---|---|
| 0%-20% | 61 | +0.0005 |
| 40%-60% | 60 | +0.0016 |
