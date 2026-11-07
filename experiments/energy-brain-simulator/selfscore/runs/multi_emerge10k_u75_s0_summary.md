# Run emerge10k_u75_s0 at a glance

**Generation 0 of 3000** | epoch 0 | 0.2 min | 0 total wipes so far

**Headline:** the best brain averages **53%** accuracy over all 24 problems (15% once chance is subtracted); solves **1** at 90%+.

## Epochs (a new epoch starts after every wipe)

| epoch | from gen | to gen | tests | start | score when it ended | ended by |
|---|---|---|---|---|---|---|
| 0 | 0 | running | 24 problems | all random | 0.451 | still running |

## Best brain, every problem (fixed reference test data; each # is 10% accuracy; chance-adjusted = how far above guessing)

| problem | accuracy | | chance-adjusted | in this epoch's tests |
|---|---|---|---|---|
| parity4 | 50% | `#####.....` | +0% | yes |
| parity6 | 47% | `#####.....` | -6% | yes |
| majority5 | 66% | `#######...` | +31% | yes |
| mux6 | 55% | `#####.....` | +9% | yes |
| adder2 | 25% | `##........` | +12% | yes |
| quadrants | 48% | `#####.....` | -4% | yes |
| moons | 92% | `#########.` | +84% | yes |
| circles | 67% | `#######...` | +34% | yes |
| spirals | 56% | `######....` | +11% | yes |
| blobs3 | 46% | `#####.....` | +20% | yes |
| parity3of6 | 52% | `#####.....` | +3% | yes |
| cmp3 | 70% | `#######...` | +41% | yes |
| addmod4 | 19% | `##........` | -8% | yes |
| and_or10 | 66% | `#######...` | +32% | yes |
| majority7 | 63% | `######....` | +27% | yes |
| checker3 | 55% | `#####.....` | +9% | yes |
| sine | 39% | `####......` | -21% | yes |
| blobs4 | 23% | `##........` | -3% | yes |
| rings3 | 35% | `###.......` | +2% | yes |
| xorgauss | 49% | `#####.....` | -1% | yes |
| moons_noisy15 | 87% | `#########.` | +74% | yes |
| xor2 | 50% | `#####.....` | +0% | yes |
| parity3 | 62% | `######....` | +25% | yes |
| xor2_d4 | 39% | `####......` | -22% | yes |
