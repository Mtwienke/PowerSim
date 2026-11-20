# Gao reproduction clarification checklist

Draft only; **not sent**. Context and immutable source links:
[public-example audit](gao-source-audit.md).

If contacting the authors, use a publicly supplied correspondence channel
(the journal lists corresponding author Juan Yu). Do not open a public issue,
send email or upload our data without a separate user decision.

Suggested request:

> We are reimplementing your TNNLS paper, DOI 10.1109/TNNLS.2024.3437741.
> We found the AI4POWER public examples at revision e775ce3 and would like to
> distinguish their settings from the final-paper experiments. Could you share
> the final project/data, or clarify the following points?
>
> 1. What is the exact renewable-penetration formula? What are the total and
>    per-site wind/PV capacities and PQ-bus locations for IEEE-30's
>    35.58%, 40.71% and 47.24% cases? Is the denominator total installed
>    conventional plus renewable capacity? A 335 MW conventional total with
>    185/230/300 MW renewables matches those rounded percentages, but we do not
>    want to assume that this was your construction.
> 2. Could you supply the data-generation script, electrical case modifications,
>    wind/PV power curves, exact load/weather sampling and treatment of failed
>    OPF solves? Are the stated 10,000 samples successful solves or attempts?
> 3. Are JSON `pd` and `pg` in per-unit on a 100 MW base, as indicated by
>    `violation_calculation.py`? Is `coff` the derivative in those same units?
>    Was any further scaling/centering done before writing the JSON?
> 4. Which settings produced Table II: the journal's 0.01 learning rate and
>    squared L0/L2/L4/L5, or the examples' Adam 0.001 with decay 2e-6,
>    MAE prediction/balance/branch/exploration terms and batch 50?
> 5. Was the examples' sensitivity mask (`sum(coff) != 0`, `max(coff) < 10`)
>    used in the final experiments? Is that a signed maximum or maximum
>    absolute value? Does the masked loss divide by all entries or valid ones?
> 6. How was the generator-bound L3 term implemented in the final model?
>    The public IEEE-30 M4/M5 updates appear to omit it and apply the branch
>    derivative loss only to branch row 20. Were other active constraints used?
> 7. Which hidden feature feeds the decoder: the second of three hidden
>    layers, or the last hidden layer immediately before the output?
>    The paper says two decoder hidden layers, while the public IEEE-57
>    decoder has three 200-unit ReLU layers and batch 500. What were the
>    final architecture, checkpoint selection and optimizer settings?
> 8. What per-sample reconstruction norm and physical conversion define the
>    6/7 MW gate? How was its threshold chosen? Is generation accuracy the
>    fraction of individual generator errors <1 MW, or another reduction?
> 9. Does Fig. 6 use the public evaluator's total branch-scenario event count
>    beyond RATE_A + 1 MW after slack rebalance? Were branch ratings otherwise
>    the unmodified PYPOWER case values?
>
> The README's Baidu share did not expose files through our anonymous access.
> A working archive, alternative link or even the generator plus a few JSON
> records (`pd`, `pg`, `coff`, `info`) would be very helpful. We have preserved
> negative results and are not claiming numerical reproduction.
