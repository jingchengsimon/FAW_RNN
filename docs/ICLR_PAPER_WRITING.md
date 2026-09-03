# ICLR Paper Writing and Interpretation Constraints

This document is the reusable source of truth for polishing the GaWF ICLR paper, including the
Background, Abstract, Results, captions, and Conclusion. Read it before changing manuscript prose.
User instructions attached to a specific draft remain higher priority. Numerical values below are
the current confirmed paper values; if a retained structured result changes, verify the new result
before updating the prose rather than silently reconciling a discrepancy.

## Response and output format

- Discuss edits with the user in Chinese; write manuscript prose and captions in academic English.
- Treat Chinese comments embedded in an English draft as binding edit instructions and remove them
  from the polished output.
- When the user requests raw LaTeX, return one copyable fenced `latex` block.
- Preserve valid, compilable LaTeX. Use forms such as
  `\(\Delta \mathrm{Acc}_{\mathrm{location}}=-2.5\,\mathrm{pp}\)` and never malformed
  subscripts, `\~`, or nested parentheses.
- Do not invent data, citations, analysis definitions, causal claims, or figure panels.
- Avoid rhetorical questions in Results. Introduce the next analysis declaratively, for example,
  `We next determined which connections remained open...`.
- Default to direct semantic succession. Unless the user explicitly requests one, do not add
  discourse markers that impose a contrast, transition, or summary, such as `however`,
  `nevertheless`, `in contrast`, `together`, `overall`, `collectively`, `therefore`, or `thus`.
  Preserve markers supplied by the user unless the user asks to remove them.

## Results prose structure

For each substantive Results paragraph:

1. Lead with the conclusion.
2. Describe direction and degree qualitatively in the sentence (`more negative`, `weaker`,
   `near baseline`, `more pronounced`).
3. Put exact values in a following parenthesis with explicit labels. Do not rely on value order.
4. End with an interpretation using `These results suggest`, `reveal`, `indicate`, `confirm`, or
   `demonstrate`.

Do not use multiplier comparisons such as `three times larger`, `four times as far`, or `five to
ten times higher`. Replace them with qualitative degree language and labeled parenthetical values.
Method/setup paragraphs need not force a conclusion-first pattern, but they should state why the
analysis is necessary and how it advances the section's argument.

Results within a section may form a progressive evidence chain rather than parallel observations.
Keep that order explicit through the content and sentence order. Add transition or summary markers
only when the user explicitly requests them.

Captions should state the panel layout, analysis definition, central result, inference unit, and
essential caveats without duplicating every value from the main text.

## Fixed terminology

| Intended meaning | Required wording |
|---|---|
| Location and identity together | `task variables` |
| Conceptual task variables | `target identity`, `target location` |
| First-mention glosses | `target identity (digit class)`, `target location (spatial sector)` |
| Discrete analysis conditions | `digit conditions`, `sector conditions` |
| Tuning axes | `digit tuning`, `sector tuning` |
| Condition-selected hidden units | `digit-tuned units`, `sector-tuned units` |
| Condition-selected ensembles | `digit-tuned ensembles`, `sector-tuned ensembles` |
| Analysis labels | `target-identity analysis`, `target-location analysis` |
| Information content | `location information`, `identity information` |
| Outputs | `location readout`, `identity readout` |
| Connection classes | `input pathway`, `recurrent pathway` |
| Units responsive to multiple variables | `mixed selectivity` |

Do not use `features` or `streams` as an umbrella term for location and identity. `Input and
recurrent pathways` may be used as parallel labels for the two connection classes. Use
`identity-relevant` only for a conceptual interpretation; use `digit-tuned` for sets defined by
the digit tuning procedure. Keep semantic levels parallel: do not present `digit` and `location`,
or `identity` and `sector`, as a pair of task variables. Use `target identity` and `target
location` at the task-variable level, `digit conditions` and `sector conditions` at the condition
level, and `digit tuning` and `sector tuning` at the tuning level.

For recurrent groups, source precedes destination in `\(T\!\to\!T\)`, `\(T\!\to\!R\)`,
`\(R\!\to\!T\)`, and `\(R\!\to\!R\)`. The notation is condition-specific: `T` is the ensemble
tuned to the current digit or sector, and `R` is its remainder rather than a biologically
established untuned population.

`\(\dg\)` is the gate value for one connection under one condition minus that connection's own
mean across all conditions of the same task variable. Explain it as a within-connection baseline,
not a comparison between raw gate values at different locations or connections.

## Paper premise and GaWF architecture

The paper addresses a gap between cortical feedback and standard recurrent architectures.
Conventional RNNs omit feedback from later processing stages. LSTM and GRU gates are computed from
feedforward input and local hidden state and act at unit level, so they lack feedback-dependent,
connection-specific modulation. S5 and Mamba improve sequence modeling but retain channel-level
control and do not return the model's own output as feedback.

The biological motivation is dendritic computation. Feedforward driver inputs preferentially
target proximal dendrites, whereas top-down modulatory inputs target distal apical dendrites.
Their nonlinear interaction can alter stimulus selectivity and behaviorally relevant processing,
providing a substrate for input-specific rather than cell-wide modulation.

GaWF uses the detached previous outputs of two readout heads as feedback. The feedback vector is
the concatenation of nine location logits and ten identity logits (dimension `19`). Low-rank
source/destination factors `\(U\)` and `\(V\)` transform feedback into input- and recurrent-gate
matrices, and sigmoid gates modulate individual weights element-wise.

The feedback construction contains no explicit location-identity conjunction component, and each
pre-sigmoid gate is linear in this feedback. GaWF's low gate interaction variance is therefore
partly constrained by design. The learned result is the complementary assignment of location and
identity to the input and recurrent pathways. This assignment is localized to the source-side
factor `\(U\)`, but variance alignment alone does not establish causal order.

## Task and behavioral results

The Cluttered Tracking MNIST task presents moving targets and distractors. The model predicts the
target's digit identity and its location in a `\(3\times3\)` sector grid. Six architectures are
matched at approximately `\(0.587\)`M parameters and trained with ten seeds.

- GaWF outperforms RNN, LSTM, GRU, S5, and Mamba on both readouts.
- GaWF test accuracy: location `\(93.3\)`, identity `\(86.6\)`.
- Closest baseline Mamba: location `\(92.6\)`, identity `\(83.1\)`.
- Every architecture predicts location more accurately than identity.
- Both readouts fall to chance at a target-assignment switch. GaWF falls less and recovers earlier.
- Feedback ablation reveals a one-way dependence: location feedback supports both task variables,
  whereas identity feedback has a more selective influence. This is not a clean double
  dissociation.
- For location accuracy, digit-segment shuffling has little effect and sector-segment shuffling
  causes a large reduction (`\(\Delta \mathrm{Acc}_{\mathrm{location}}=-2.5\,\mathrm{pp}\)`
  and `\(-31.1\,\mathrm{pp}\)`, respectively).
- Identity accuracy is also more impaired by sector-segment than digit-segment shuffling
  (`\(-35.5\,\mathrm{pp}\)` and `\(-16.3\,\mathrm{pp}\)`, respectively).

Figure 1 is fixed as top `Location`, bottom `Identity`. Its four columns show test accuracy,
validation loss, target-switch recovery, and feedback ablation.

## Dynamic sparse routing

Both gates have U-shaped distributions with values concentrated near `0` and `1`. The input gate
is biased toward the closed state, whereas the recurrent gate is more evenly divided between the
two endpoints.

- Input gate: `\(G<0.1\)`, `\(64.0\%\)`; `\(G>0.9\)`, `\(21.2\%\)`.
- Recurrent gate: `\(G<0.1\)`, `\(32.0\%\)`; `\(G>0.9\)`, `\(34.8\%\)`.

Multiplication by the gate increases the mass of `\(G\odot W\)` near zero while preserving a
subset close to its original magnitude. Describe this as `feedback-dependent dynamic sparse
routing`: feedback suppresses much of the available connectivity while transmitting a selected
subnetwork near full strength on each frame.

Figure 3 is a one-by-four layout: input-gate distribution, recurrent-gate distribution, input
weights, and recurrent weights. Do not refer to top or bottom rows.

## Task-variable demixing

Use `sector-by-digit interaction` on first mention and define it as non-additive dependence on
specific location-identity combinations. A representation is `demixed` when one task variable
dominates with a small interaction component; it is `conjunctive` when interaction accounts for a
substantial share.

Activation decomposition:

- Encoder: location `\(51.3\%\)`, identity `\(6.7\%\)`, interaction `\(42.0\%\)`.
- Hidden activation: identity `\(52.5\%\)`, location `\(31.3\%\)`, interaction
  `\(16.2\%\)`.

Interpret the shared encoder as producing a location-biased but conjunctive representation and
recurrent processing as shifting the representation toward identity.

Synapse-level GaWF gates:

- Input gate: location `\(76.7\%\)`, identity `\(16.0\%\)`, interaction `\(7.3\%\)`.
- Recurrent gate: identity `\(71.7\%\)`, location `\(23.3\%\)`, interaction `\(5.0\%\)`.

Destination-unit comparison:

- Projected GaWF input gate: location `\(88.2\%\)`.
- Projected GaWF recurrent gate: identity `\(86.5\%\)`.
- LSTM input gate: identity `\(52.3\%\)`.
- LSTM forget/output gates: location `\(62.6\%\)` / `\(65.9\%\)`.
- GRU reset/update gates: location `\(56.9\%\)` / `\(51.8\%\)`.

Conventional gates reflect the same task variables, but their task-variable dominance is less
pronounced. Do not frame greater conventional-gate interaction variance as the central contrast.
The comparison cannot isolate connection-level granularity from the feedback-derived control
signal.

Figure 4 layout:

- Top left: GaWF synapse-level gates.
- Top right: encoder and hidden activations.
- Bottom left: destination-unit projections of GaWF gates.
- Bottom centre: LSTM gates.
- Bottom right: GRU gates.

Current file:
`results/save/iclr_figs/Fig4_gate_task_variable_specialization_2x3_10seed.pdf`.

## Input-gate spatial selection

This section follows the demixing analysis. It begins from the result that encoder activation and
input-gate modulation are both dominated by target location, then identifies which input
connections the gate selects. Use `\citep{desimone1995neural}` only to motivate the selective
attention hypothesis, not to imply that prior work established the present synaptic mechanism.

The CNN output has 32 feature channels on a common `\(6\times6\)` spatial grid. Each grid
position corresponds to an approximate receptive field in the original frame. Sector-conditioned
activation peaks move with target location, whereas digit-conditioned maps show no corresponding
spatial displacement. Location is therefore represented primarily by the spatial arrangement of
encoder activation.

Input `\(\dg\)` maps show a compact region of increased modulation that follows the target sector
and aligns with the encoder-activation peak. Interpret this as preferential opening of connections
from location-aligned encoder positions.

Methods/data-processing requirement: for the sector-conditioned input-gate analyses, exclude
zero-feedback sequence-reset frames and sample an equal number of frames from each sector
condition within each seed. Document this equal-`\(n\)` sampling and reset-frame exclusion in
Methods rather than repeating it in the Figure 6 caption.

Location-matched sign result:

- Positive weights: `\(\overline{\dg}_{W>0}=+0.276\)`.
- Negative weights: `\(\overline{\dg}_{W<0}=+0.289\)`.
- Sign gap: `\(\overline{\dg}_{W>0}-\overline{\dg}_{W<0}=-0.014\)`.
- All matching connections: `\(\overline{\dg}=+0.283\)`.

Both signs receive similar increases relative to their own baselines. Within the matching group,
positive-weight modulation approaches zero as magnitude increases, whereas negative-weight
modulation remains nearly constant (`\(\beta_{W>0}=-0.043\)`,
`\(\beta_{W<0}=+0.007\)`). Other spatial sources are only weakly modulated.

Figure 6 layout:

- Left: sector-conditioned encoder maps.
- Centre: sector-conditioned input `\(\dg\)` maps.
- Right upper: matching-source `\(\dg\)` versus `\(\lvert W\rvert\)`.
- Right lower: other-source `\(\dg\)` versus `\(\lvert W\rvert\)`.

Use `\label{fig:inputgate}`. Current file:
`results/save/iclr_figs/Fig6_overall_sector_input_gate_1x3_10seed.pdf`.

## Recurrent-gate disinhibition

This section also follows the demixing analysis and must present one progressive argument:

1. Localize recurrent `\(\dg\)`.
2. Establish sign asymmetry within the selected group.
3. Show how that asymmetry depends on connection magnitude.
4. Confirm its functional direction with gate-dependent recurrent current.

Start from the result that hidden activation and recurrent-gate modulation are dominated by target
identity. Recurrent units have mixed selectivity, so define digit-tuned sets separately within each
seed. For each digit, `\(T_d\)` is the top `\(10\%\)` of a common eligible pool by standardized
marginal digit tuning (`\(\lvert T_d\rvert=23\text{--}25\)` of 256 units). Sets may overlap across
digits. The remainder for the current digit is `R`. Tuning uses held-out validation activations;
gate and current analyses use an independent test split.

Recurrent `\(\dg\)` localization:

- Digit conditions: `\(T\!\to\!T=-0.310\)`, `\(T\!\to\!R=-0.094\)`,
  `\(R\!\to\!T=-0.005\)`, `\(R\!\to\!R=+0.014\)`.
- Sector conditions: `\(T\!\to\!T=-0.095\)`, `\(T\!\to\!R=-0.079\)`.

The main conclusion is that identity-dependent recurrent modulation concentrates within the
digit-tuned ensemble. Sector is an auxiliary comparison and must not displace that conclusion.

Within digit-conditioned `\(T\!\to\!T\)`, both signs receive negative modulation, but
negative-weight connections receive more negative modulation
(`\(\overline{\dg}_{W>0}-\overline{\dg}_{W<0}=+0.110\)`). This ordering is consistent with
disinhibition because reducing the gate on a negative-weight connection reduces its negative
current contribution.

Positive-weight modulation approaches zero with increasing `\(\lvert W\rvert\)`, while
negative-weight modulation remains nearly constant (`\(\beta_{W>0}=+0.129\)`,
`\(\beta_{W<0}=-0.021\)`). Relate this pattern to the input-gate magnitude result.

Gate-dependent current confirms the mechanism in digit-conditioned `\(T\!\to\!T\)`:

- Positive-weight contribution: `\(\Delta I^{\mathrm{gate}}_{W>0}=-0.013\)`.
- Negative-weight contribution: `\(\Delta I^{\mathrm{gate}}_{W<0}=+0.056\)`.
- Balanced contribution: `\(\Delta I^{\mathrm{gate}}_{\mathrm{bal}}=+0.025\)`.
- Sector-conditioned balanced contribution: `\(+0.002\)`.

The balanced bar is the connection-count-weighted mean of the two sign classes, not their sum.
The release from negative input exceeds the reduction in positive input, producing a positive net
per-connection current change within the tuned ensemble.

GaWF is not Dale-constrained. `Disinhibition` is a functional connection-level description of
reduced negative recurrent contribution, not evidence for an identified inhibitory cell class or
biological interneuron circuit.

Figure 7 layout:

- Top left: Digit `\(\dg\)` bars.
- Top centre: Sector `\(\dg\)` bars.
- Top right upper/lower: Digit/Sector `\(T\!\to\!T\)` magnitude curves.
- Bottom left/right: Digit/Sector connection-normalized `\(\Delta I^{\mathrm{gate}}\)` bars.

Use `\label{fig:recurrentdisinhibition}`. Current file:
`results/save/iclr_figs/Fig7_recurrent_gate_disinhibition_and_current_2x3_10seed.pdf`.

Figure 7 summary statistics must remain accurate:

- Top bar dots are individual training seeds; error bars are cross-seed SEM.
- Top curves are quantile-binned pooled connection means plus/minus SEM.
- Bottom gray dots are condition means averaged across seeds, not individual seeds.
- Bottom bars/error bars summarize ten seed-level condition averages.

Methods/statistical-testing requirement: top-bar inference uses exact two-sided sign-flip tests
against zero over the ten seed-level statistics. Bottom-bar inference uses two-sided one-sample
`\(t\)`-tests against zero over the ten seed-level condition averages. The displayed stars use an
uncorrected `\(p<0.05\)` threshold. Record these test details in Methods rather than repeating
them in the Figure 7 caption.

## Abstract and Conclusion

The Abstract follows:

`problem -> architectural gap -> proposed model and method -> main results -> significance/outlook`

It must include four findings:

1. GaWF outperforms matched recurrent and state-space baselines.
2. GaWF gates implement dynamic sparse routing of effective connectivity.
3. Input and recurrent gates specialize for complementary task variables.
4. Input feedback selects spatially aligned connections, while recurrent feedback disinhibits
   digit-tuned ensembles.

The Conclusion should state that GaWF transforms one shared feedback signal into pathway-specific
connection modulation: spatial selection in the input pathway and functional disinhibition in the
recurrent pathway. Present connection-level feedback gating as a complementary inductive bias to
conventional unit-level gating. End by acknowledging that GaWF is a computational abstraction
tested on a controlled tracking task and that future work should examine richer feedback sources,
deeper recurrent hierarchies, and broader context-dependent tasks.

## Citation keys

The current Background and Results bibliography uses:

`chance2002gain`, `yang2016dendritic`, `letzkus2011disinhibitory`,
`larkum2013cellular`, `fisek2023corticocortical`, `smith2013dendritic`,
`lavzin2012nonlinear`, `takahashi2016active`, `takahashi2020active`,
`petreanu2009subcellular`, `shen2022distinct`, `kording2001supervised`,
`guerguiev2017towards`, `payeur2021burst`, `richards2019deep`,
`lillicrap2020backpropagation`, `hochreiter1997long`, `cho2014properties`,
`chung2014empirical`, `poirazi2003pyramidal`, `beniaguev2021single`,
`poirazi2020illuminating`, `smith2023simplified`, `gu2024mamba`, and
`desimone1995neural`.

Prefer references already present in the proposal. If a new citation is necessary, verify the
primary paper or official publisher page and supply a complete BibTeX entry. A citation may support
biological or computational motivation; it must not be used to imply that prior work established a
new GaWF result.

## Final polishing checklist

Before returning revised prose:

- Identify the paragraph's single main conclusion and its position in the section's evidence chain.
- Preserve every user-specified direction, value, grouping, and interpretation.
- Apply conclusion-first, qualitative-description, labeled-values, interpretation-last structure.
- Verify every panel reference against the current merged figure layout.
- Enforce the fixed `task variables`, condition, pathway, and tuned-set terminology.
- Remove multiplier comparisons and unsupported causal language.
- Preserve the no-Dale-constraint caveat whenever interpreting recurrent sign as disinhibition.
- Return the complete revised passage, not only isolated suggestions, unless the user asks for
  discussion only.
