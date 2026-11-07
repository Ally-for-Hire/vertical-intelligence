# probnet constants (open problems: none of these is derived from first principles)

| Name | Where | Value | Status |
| --- | --- | --- | --- |
| majority | all nets | 0.55 | GIVEN by the user |
| n_min | snarc.py | 10 | ARBITRARY: visits before a failing cell may split / a child may be judged |
| depth | snarc.py | 6 | ARBITRARY: how many halvings of the radius are allowed (min radius = r0 / 64) |
| cap | snarc.py | 600 | ARBITRARY: maximum cells |
| step size | snarc.py | 1/(visits+2) | DERIVED: running average (no learning-rate knob) |
| vote weight | snarc.py | 1/radius | ARBITRARY: specific cells outweigh general cells |
| 2 * se | snarc.py merge test | 2 | conventional significance, not tuned |
| trials per problem | run_suite.py | 8000 (125 batches x 64) | ARBITRARY budget |
| car success signal | car_task.py | on road and (lateral error not growing or < 0.25 half-width) | ARBITRARY task definition |
| radius r0 | snarc.py | sqrt(D)/2 | DERIVED: reaches every corner of the unit cube from the centre |
| lr, wd, prune_eps, hid scale ... | net.py net2.py grid.py | various | ARBITRARY, from superseded attempts 0-3 |
| alpha | snarc_rel.py | 0.01, spent as alpha/(k(k+1)) over looks | conventional level; spending rule is standard |
| every / log | snarc_rel.py | 400 trials / 800-trial log | ARBITRARY |
| drag factor | drift_test.py | 2.5 | ARBITRARY (turned out not to matter) |
