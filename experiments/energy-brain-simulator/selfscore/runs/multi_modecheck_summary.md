# Run modecheck at a glance

**Generation 50 of 51** | epoch 0 | 1.0 min | 0 total wipes so far

**Headline:** the best brain averages **53%** accuracy over all 24 problems (16% once chance is subtracted); solves **0** at 90%+.

## Epochs (a new epoch starts after every wipe)

| epoch | from gen | to gen | tests | start | score when it ended | ended by |
|---|---|---|---|---|---|---|
| 0 | 0 | running | 24 problems | all random | 0.490 | still running |

## Best brain, every problem (fixed reference test data; each # is 10% accuracy; chance-adjusted = how far above guessing)

| problem | accuracy | | chance-adjusted | in this epoch's tests |
|---|---|---|---|---|
| parity4 | 62% | `######....` | +25% | yes |
| parity6 | 50% | `#####.....` | +0% | yes |
| majority5 | 56% | `######....` | +12% | yes |
| mux6 | 58% | `######....` | +16% | yes |
| adder2 | 25% | `##........` | +12% | yes |
| quadrants | 58% | `######....` | +16% | yes |
| moons | 87% | `#########.` | +75% | yes |
| circles | 66% | `#######...` | +31% | yes |
| spirals | 56% | `######....` | +13% | yes |
| blobs3 | 39% | `####......` | +9% | yes |
| parity3of6 | 53% | `#####.....` | +6% | yes |
| cmp3 | 47% | `#####.....` | -6% | yes |
| addmod4 | 31% | `###.......` | +8% | yes |
| and_or10 | 64% | `######....` | +27% | yes |
| majority7 | 65% | `######....` | +30% | yes |
| checker3 | 55% | `######....` | +10% | yes |
| sine | 39% | `####......` | -21% | yes |
| blobs4 | 26% | `###.......` | +2% | yes |
| rings3 | 37% | `####......` | +6% | yes |
| xorgauss | 56% | `######....` | +11% | yes |
| moons_noisy15 | 85% | `#########.` | +71% | yes |
| xor2 | 50% | `#####.....` | +0% | yes |
| parity3 | 62% | `######....` | +25% | yes |
| xor2_d4 | 50% | `#####.....` | +0% | yes |

## What the random cull did (change in average accuracy, bucketed by that generation's cull draw)

| cull draw | generations | avg change in accuracy |
|---|---|---|
| 0%-20% | 41 | +0.0002 |
| 20%-40% | 4 | +0.0012 |
| 40%-60% | 5 | +0.0051 |
| 60%-80% | 1 | +0.0068 |
