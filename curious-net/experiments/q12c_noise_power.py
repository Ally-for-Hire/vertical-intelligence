"""Q12c: detection power of the lack-of-fit test vs measurement noise (same worlds as q12b, n_main = 16, 150 worlds per cell, alpha = 0.01).
If the library can imitate the true law within the noise, power is low; lowering the noise should raise it."""
import numpy as np
import q12_law_discovery as Q
import q12b_out_of_library as B

rng = np.random.default_rng(5)
B.rng = rng
inlib = [("inlib", float(np.exp(rng.uniform(0, np.log(25)))), [np.inf, 40.0, 15.0, 6.0][int(rng.integers(4))]) for _ in range(150)]
stokes = [("stokes", float(np.exp(rng.uniform(np.log(3), np.log(25)))), float(rng.choice([0.15, 0.4, 1.0, 3.0]))) for _ in range(150)]
varg = [("varg", float(np.exp(rng.uniform(np.log(3), np.log(25)))), float(rng.choice([15.0, 30.0, 60.0]))) for _ in range(150)]
print("relative noise / absolute noise (s) | flag rate (alpha=0.01): in-library (false alarm), stokes (k 0.15-3), varg (R 15-60 m)")
for rel, ab in ((0.03, 0.05), (0.01, 0.02), (0.003, 0.005), (0.001, 0.002)):
    Q.NOISE_REL, Q.NOISE_ABS = rel, ab
    B.Q.NOISE_REL, B.Q.NOISE_ABS = rel, ab
    out = []
    for worlds in (inlib, stokes, varg):
        out.append(np.mean([B.run(w, 16)[5] < 0.01 for w in worlds]))
    print(f"   {rel:5.3f} / {ab:5.3f}   | {out[0]:.2f}   {out[1]:.2f}   {out[2]:.2f}")
