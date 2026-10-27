---
title: What works: expertise, learning to learn, transfer and motivation (research check, 2026-10-18)
domain: neuroblock-research
kind: research
tags: [expertise, transfer, motivation, curiosity, learning-science, meta-learning, genius]
updated: 2026-10-18
---

# What works: expertise, learning to learn, transfer and motivation

## Summary

The user believes (a) all problems intrinsically relate, so mastering one domain teaches how to think and speeds up learning anything else; (b) genius is motivation plus processing speed, with motivation the big one;
and wants to "work from what works" (including books about making geniuses). This page records a quick, sourced check of the research behind those ideas. **Method limits:** web-search summaries only (abstracts and secondary pages),
not full papers and not the books. Items marked *unverified* come from memory or from a secondary source and need checking before anyone relies on them.

## What the evidence supports

* **Expertise is chunked, long-term knowledge that makes known things nearly automatic.** Chase & Simon's chunking account of chess expertise (1973) was revised by Gobet & Simon's *template theory* (experts recall more than pure chunking predicts; templates in long-term memory).
  Cognitive load theory (Sweller) says the same from the learning side: schemas are treated as one item by a small working memory, and repeated practice *automates* them, which frees capacity; automation develops only for aspects that stay consistent across situations.
  Sources: [Gobet & Simon, Brunel repository](https://bura.brunel.ac.uk/handle/2438/815?mode=full), [overview of cognitive load theory](https://ro.uow.edu.au/edupapers/136) (secondary).
* **"Learning to learn" is real, but tied to shared structure.** Harlow (1949) showed monkeys solving new object-discrimination problems in fewer trials after many earlier problems of the same kind ("learning set"); DeepMind (2018) reproduced the effect with a meta-reinforcement-learning network.
  The effect appears when the problems share a common structure. Sources: [Meta-Learned Models of Cognition](https://arxiv.org/pdf/2304.06729), [DeepMind, prefrontal cortex as a meta-RL system](https://deepmind.google/discover/blog/prefrontal-cortex-as-a-meta-reinforcement-learning-system/).
* **Curiosity as *learning progress*.** Oudeyer & Kaplan's intrinsic-motivation systems steer an agent toward what yields the most learning progress (neither already predictable nor hopelessly unpredictable); stages of development emerged without scripting;
  a human study reported people following learning progress when exploring activities of mixed difficulty. Schmidhuber's formal theory frames it as *compression progress*: reward for getting better at predicting or compressing. Sources: [Oudeyer, Kaplan & Hafner 2007](https://adapt.informatik.hu-berlin.de/pub/papers/ims.pdf), [Schmidhuber 2010 overview](https://www.ece.uvic.ca/~bctill/papers/ememcog/Schmidhuber_2010.pdf).
* **Motivation has kinds, not just amounts.** Self-determination theory: autonomy, competence and relatedness as basic needs; extrinsic pay can undermine existing intrinsic interest (Deci, 1971). Source: [Ryan & Deci 2022 encyclopedia entry](https://selfdeterminationtheory.org/wp-content/uploads/2023/01/2022_RyanDeci_SDT_Encyclopedia.pdf). I did not find a meta-analysis of SDT in this pass.
* **Study techniques with strong support:** practice testing (retrieval) and spaced (distributed) practice were rated "high utility" by Dunlosky et al. (2013); interleaving, elaborative interrogation and self-explanation "moderate"; rereading, highlighting, summarizing "low". Sources: [APS summary](https://www.psychologicalscience.org/news/releases/which-study-strategies-make-the-grade.html) (secondary).
  Bjork's "desirable difficulties" (spacing, interleaving, testing, variation; the fluency illusion) is the framework, see [overview](https://www.psychologicalscience.org/?p=128850) (secondary).
* **Productive failure** (attempting problems before instruction) has a 53-study meta-analysis (Sinha & Kapur) reported to favour it, more for secondary and undergraduate students, needing some prior knowledge. Only press coverage was read; headline claims ("twice as efficient", "threefold") are promotional framing, *unverified*. Source: [WEF write-up](https://www.weforum.org/stories/2021/09/students-who-productively-fail-learn-more/).

## What the evidence challenges

* **Far transfer ("good at X makes you better at everything") is weakly supported.** Sala & Gobet's meta-analyses of chess, music instruction and working-memory training in children found small pooled effects that shrank toward zero in well-designed studies (active controls); a second-order meta-analysis found
  far-transfer effects small or null after controlling for placebo and publication bias. (These are the skeptics' own analyses; critiques were not in the results.) Barnett & Ceci's taxonomy: far transfer is rare but can occur, and may be predictable from how the new context differs along content and context dimensions.
  Sources: [Sala & Gobet 2017](https://pmc.ncbi.nlm.nih.gov/articles/PMC5724589), [LSE repository](https://researchonline.lse.ac.uk/id/eprint/102204), [Barnett & Ceci overview](https://www.globalcognition.org/transfer-of-learning/) (secondary).
* **Practice is not everything.** Macnamara, Hambrick & Oswald (2014) meta-analysis: deliberate practice explained about 26% of performance differences in games, 21% in music, 18% in sports, 4% in education and under 1% in professions; critics say it used a broader definition of "deliberate practice" than Ericsson's.
  Sources: [Macnamara 2014 summary](https://www.sciencedaily.com/releases/2014/07/140716123835.htm), [Frontiers critique 2019](https://pmc.ncbi.nlm.nih.gov/articles/pmid/31708836). This fits "motivation drives practice" only partly: practice amount is one factor among others.
* **Growth-mindset interventions have weak effects overall** (Sisk et al. 2018: two meta-analyses, weak for both; possibly larger for low-SES or at-risk students). The commonly quoted d about 0.08 is from secondary sources, *unverified*. Source: [Sisk et al. 2018 PDF](https://hhs.purdue.edu/skill-learning-and-performance-lab/wp-content/uploads/sites/43/2024/08/sisk-et-al-2018-to-what-extent-and-under-which-circumstances-are-growth-mind-sets-important-to-academic-achievement-two.pdf).
* **"Genius = motivation + processing speed"** was *not checked* in this pass (no search on intelligence, processing speed or creativity research). Treat it as the user's hypothesis.

## Books

None of the books were read. The research behind several was checked above (Ericsson's deliberate practice via the Macnamara critique; Bjork; Dunlosky; Sweller). From memory only, *unverified here*: *Make It Stick* (retrieval, spacing, interleaving), Dehaene's *How We Learn*, Epstein's *Range*
(breadth and analogical transfer). Popular claims in the genre (10,000 hours, mindset, grit) are the weakest-supported parts per the above and should not be built in without checking.

## What it means for the simulator (proposed)

1. **Test "all problems relate" directly instead of assuming it.** Build problem families whose *shared structure is controllable*, and measure transfer as a function of the overlap. Predicted by the research: large speed-up when structure is shared, little when it is not.
2. **Motivation as learning progress per unit of energy** (Oudeyer / Schmidhuber): the Self chooses what to work on by where its uncertainty is falling fastest, which is also the earlier principle "reward effort only while uncertainty is dropping".
3. **Store strategies explicitly** (the "how it thought" part), since learning-to-learn shows up when tasks share structure.
4. **Consolidation as chunking/templates:** merge many specific neurons into a general template; schedule retrieval (replay) of stored recipes at spaced intervals to keep them automatic.

## Open

Verify against full texts before relying on: effect sizes cited above from secondary pages; the Kapur claims; Barnett & Ceci details; whether motivation or speed (or neither) is the dominant factor in exceptional performance (not searched).
