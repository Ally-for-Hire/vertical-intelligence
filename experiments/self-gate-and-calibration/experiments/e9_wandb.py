"""E9c W&B view (author's standing rule: every experiment readable by non-specialists). Reads the two E9 result files and logs the data with its sources,
the fits and the verdicts. Usage: python e9_wandb.py   (after SIDEREAL=1 python e9_kepler.py and SIDEREAL=1 MOONS=NSSDC python e9_kepler.py)"""
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vi"))
from track import Tracker

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
jpl = json.load(open(os.path.join(R, "e9_kepler_sidereal.json"))); nss = json.load(open(os.path.join(R, "e9_kepler_sidereal_nssdc.json")))
ARMS = {"JPL": ("Moon periods from the JPL table", "The four Galilean moon periods as printed in the JPL satellite mean-elements table (Io 1.762732 days, Europa 3.525463 days)."),
        "NSSDC": ("Moon periods from the NASA fact sheet", "The same moons with the sidereal periods printed in the NASA NSSDC Jovian Satellite Fact Sheet (Io 1.769138 days, Europa 3.551181 days).")}
tk = Tracker("E9c", "Rediscovering Kepler's third law from real NASA data", "Can a computer, given only measured orbit sizes and periods, find that period grows as size to the power 1.5, notice that Jupiter's moons follow a different constant than the planets, and read off the mass ratio of the Sun to Jupiter?",
             "Real measured data: the 8 planets of the Solar System (distance from the Sun, orbital period) and the 4 big moons of Jupiter (distance from Jupiter, orbital period). No noise model is given; every candidate law is fitted in log space and compared by BIC, a score that rewards fit and punishes extra free numbers. Two NASA sources disagree slightly on two moon periods, so the whole analysis is run with each.",
             ARMS, "Exponent: Kepler's law says 1.5. BIC difference: larger means 'one constant for everything' is rejected more strongly. Mass ratio: the Sun is about 1047 times heavier than Jupiter (IAU value 1047.355). Residual: how far a moon sits from the law, in percent.",
             dict(planets="NSSDC per-planet fact sheets (sidereal periods)", moon_source_a="https://ssd.jpl.nasa.gov/sats/elem/", moon_source_b="https://nssdc.gsfc.nasa.gov/planetary/factsheet/joviansatfact.html"), tags=["real data", "Kepler"])
rows = [[tk.name(k), round(d["moons_p"], 4), round(d["moons_se"], 4), round(d["dbic_hole"], 1), round(d["mass_ratio"], 1), round(d["max_abs_moon_resid_pct"], 3), round(100 * d["pluto_err"], 2)] for k, d in (("JPL", jpl), ("NSSDC", nss))]
tk.table("Results by moon data source", ["Moon data used", "Exponent fitted to the moons alone (Kepler: 1.5)", "Uncertainty", "BIC difference, one constant vs one per central body (bigger = stronger rejection)", "Implied Sun/Jupiter mass ratio (IAU: 1047.355)", "Largest moon residual, percent", "Pluto period prediction error, percent"], rows)
tk.table("Moon data side by side", ["Moon", "Distance from Jupiter (million km)", "Period in the JPL table (days)", "Period in the NASA fact sheet (days)"],
         [["Io", 0.4218, 1.762732, 1.769138], ["Europa", 0.6711, 3.525463, 3.551181], ["Ganymede", 1.0704, 7.155588, 7.154553], ["Callisto", 1.8827, 16.690440, 16.689017]])
checks = [("P271", "With the NASA fact-sheet periods the implied Sun/Jupiter mass ratio is within 2% of the IAU value 1047.35", abs(nss["mass_ratio"] / 1047.355 - 1) < 0.02, f"{nss['mass_ratio']:.1f} ({100 * (nss['mass_ratio'] / 1047.355 - 1):+.2f}%); with the JPL periods {jpl['mass_ratio']:.1f} ({100 * (jpl['mass_ratio'] / 1047.355 - 1):+.2f}%)"),
          ("P272", "The exponent fitted to the four moons alone is 1.50 +- 0.01", abs(nss["moons_p"] - 1.5) <= 0.01, f"{nss['moons_p']:.4f} +- {nss['moons_se']:.4f}"),
          ("P273", "'One constant for every orbit' is rejected by a BIC difference of at least 40", nss["dbic_hole"] >= 40, f"{nss['dbic_hole']:.1f}"),
          ("P274", "The worst moon residual is smaller with the NASA fact-sheet periods than with the JPL periods", nss["max_abs_moon_resid_pct"] < jpl["max_abs_moon_resid_pct"], f"{nss['max_abs_moon_resid_pct']:.3f}% vs {jpl['max_abs_moon_resid_pct']:.3f}%")]
tk.verdicts([(i, s, "PASSED" if ok else "FAILED", e) for i, s, ok, e in checks]); tk.finish()
for i, s, ok, e in checks: print(i, "PASSED" if ok else "FAILED", "|", e)
