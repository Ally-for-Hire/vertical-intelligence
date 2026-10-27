# neuroblock-research

Toy-simulation research project toward a brain-like system that builds its own representation of a problem, solves it from
prior context, and evolves (inspired by IFS, Society of Mind, ART, predictive processing). Code and git history live in
`~/neuroblock-research`.

Program charter: [[../vertical-intelligence/wiki/research-protocol]] (three theorems, protocol, map).

## Pages

- [[wiki/neuroblock-project-state]] - Design principles, experiment results, caveats, next steps, and how to run/reproduce.
- [[wiki/sweep-and-lab-2026-10-18]] - 9,744-run sweep findings (what matters, what lines up with the principles) and the Lab (designer, queue, bookmarks, plugins).
- [[wiki/foundations-and-energy-2026-10-18]] - the user's foundations (parts, Self, purpose, mastery), the energy economy and its first findings, the 1-thread compute limit.
- [[wiki/what-works-learning-expertise-motivation-2026-10-18]] - sourced check of expertise, learning-to-learn, transfer, curiosity and motivation research against the user's beliefs; what it means for the simulator.
- [[wiki/car-steering-test-2026-10-23]] - first car-steering grade of the default NeuroBlock + evolved policy net vs a no-block net; result inconclusive (3 seeds), code in `neuroblock-research/car/`.
- [[wiki/probnet-snarc-neuroblock-2026-10-24]] - probability-net attempts; SNARC operant conditioning + NeuroBlock cells solves the 68-problem suite to 0.893 (chance 0.570, plain net 0.965) and the car (0.85-0.91 held-out). Code: `neuroblock-research/probnet/`.
- [[wiki/car-steering-versions-2026-10-23]] - car/steering line v1-v5 (Boolean blocks, grid, the poll, cascade of trust), toy-problem ladder, first cascade results and defects.
