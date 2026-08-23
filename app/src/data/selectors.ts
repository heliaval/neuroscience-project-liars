// app/src/data/selectors.ts -- the ONLY module that touches the results JSON's raw
// shape. Components consume these typed accessors and nothing else (plans/web-app-scaffold.md
// §3.2). If the real file's shape drifts from the fixture's, it breaks here first.
import { RESULTS } from "./source";
import type { ResultsV1, PairedTest, ProvenanceEnum } from "../types/results.v1";

export type Dyad = ResultsV1["dyads"][number];
export type ExperimentId = keyof ResultsV1["experiments"];
export type Experiment<K extends ExperimentId> = ResultsV1["experiments"][K];
export type Claim = ResultsV1["tests"][string];
export type SectionId = keyof ResultsV1["meta"]["provenance"];

/** Known claim ids present in the fixture's `tests` block (plans/results-v1-fixture.md §4.10). */
export const CLAIM_IDS = [
  "exp1_above_chance",
  "h1_lodo_generalization",
  "h2_person_gain",
  "exp3_personalized_vs_population",
  "h3_dyad_gain",
  "nfi_distribution",
  "exp5_learning_curve",
  "exp6_observer_early_late",
  "exp7_observer_vs_deceiver",
] as const;
export type ClaimId = (typeof CLAIM_IDS)[number];

export function isFixture(): boolean {
  return RESULTS.meta.is_fixture;
}

export function getProvenance(section: SectionId): ProvenanceEnum {
  return RESULTS.meta.provenance[section];
}

export function getDyads(): Dyad[] {
  return RESULTS.dyads;
}

export function getDyad(pairId: string): Dyad | undefined {
  return RESULTS.dyads.find((d) => d.pair_id === pairId);
}

export function getParticipantCount(): number {
  const ids = new Set<string>();
  for (const d of RESULTS.dyads) {
    for (const p of d.participants) ids.add(p);
  }
  return ids.size;
}

export function getExperiment<K extends ExperimentId>(id: K): Experiment<K> {
  return RESULTS.experiments[id];
}

export function getClaim(id: ClaimId): Claim {
  return RESULTS.tests[id];
}

export function getPairedTest(id: ClaimId): PairedTest {
  return getClaim(id).result as PairedTest;
}

export function getAmendments() {
  return RESULTS.frozen.amendments;
}

export function getFrozenHypotheses() {
  return RESULTS.frozen.hypotheses as Record<string, string>;
}

/** Names of sections still on placeholder data, for the provenance banner. */
export function getPlaceholderSections(): SectionId[] {
  const prov = RESULTS.meta.provenance;
  return (Object.keys(prov) as SectionId[]).filter((k) => prov[k] === "placeholder");
}

export function getRealSections(): SectionId[] {
  const prov = RESULTS.meta.provenance;
  return (Object.keys(prov) as SectionId[]).filter((k) => prov[k] === "real");
}

/** Names of sections that are part real, part placeholder (e.g. the dyad roster's
 * fingerprint columns), for the provenance banner. */
export function getMixedSections(): SectionId[] {
  const prov = RESULTS.meta.provenance;
  return (Object.keys(prov) as SectionId[]).filter((k) => prov[k] === "mixed");
}

// -- exp4 condition accessors (§32 "Stranger vs Familiar", plans/frontend-stranger-vs-familiar.md
// Task 1) -- the generated type narrows `per_dyad` items to `{ [k: string]: unknown }`, so this
// is the one place that narrows and casts them, matching this file's role as the only module
// that knows the JSON's shape.

/** The three primary exp4 training conditions, in §32's STRANGER / PERSON-SPECIFIC / DYAD-SPECIFIC order. */
export const EXP4_CONDITIONS = ["universal", "person_specific", "dyad_specific"] as const;
export type Exp4ConditionId = (typeof EXP4_CONDITIONS)[number];

export interface Exp4ConditionScore {
  auroc: number;
  balanced_accuracy: number;
  f1: number;
  precision: number;
  recall: number;
  n_train?: number;
}

export interface Exp4DyadRow {
  pair_id: string;
  tested_participant: string;
  partner: string;
  n_test: number;
  held_out_seq_min: number;
  held_out_seq_max: number;
  universal: Exp4ConditionScore;
  person_specific: Exp4ConditionScore;
  dyad_specific: Exp4ConditionScore;
  n_matched: Exp4ConditionScore;
  person_other_dyad: Exp4ConditionScore;
  majority: Exp4ConditionScore;
}

export function getExp4PerDyad(): Exp4DyadRow[] {
  return getExperiment("exp4").per_dyad as unknown as Exp4DyadRow[];
}

/** One AUROC per dyad for the given condition, in `per_dyad` order. */
export function getConditionScores(cond: Exp4ConditionId): { pair_id: string; auroc: number }[] {
  return getExp4PerDyad().map((row) => ({ pair_id: row.pair_id, auroc: row[cond].auroc }));
}

export function getConditionSummary(cond: Exp4ConditionId): {
  median: number;
  nAboveChance: number;
  n: number;
  nTrainMin: number;
  nTrainMax: number;
} {
  const rows = getExp4PerDyad();
  const scores = rows.map((row) => row[cond].auroc).sort((a, b) => a - b);
  const mid = Math.floor(scores.length / 2);
  const median = scores.length % 2 === 0 ? (scores[mid - 1] + scores[mid]) / 2 : scores[mid];
  const nAboveChance = scores.filter((v) => v > 0.5).length;
  const nTrains = rows.map((row) => row[cond].n_train ?? 0);
  return {
    median,
    nAboveChance,
    n: scores.length,
    nTrainMin: Math.min(...nTrains),
    nTrainMax: Math.max(...nTrains),
  };
}

/** The single held-out test block every exp4 condition is scored against. Throws if any dyad
 * disagrees -- the section's "same held-out rows" claim would otherwise silently go false. */
export function getExp4HeldOut(): { nTest: number; seqMin: number; seqMax: number } {
  const rows = getExp4PerDyad();
  const first = rows[0];
  const held = { nTest: first.n_test, seqMin: first.held_out_seq_min, seqMax: first.held_out_seq_max };
  for (const row of rows) {
    if (row.n_test !== held.nTest || row.held_out_seq_min !== held.seqMin || row.held_out_seq_max !== held.seqMax) {
      throw new Error("exp4 per_dyad rows disagree on the held-out test block");
    }
  }
  return held;
}

export function getExp4Verdict(): string {
  return getExperiment("exp4").verdict;
}

// -- walkthrough accessors (plans/interactive-walkthrough.md Task 4). Same narrow-and-cast
// role as the exp4 block above: this file is the only module that knows the JSON's shape.
// None of these compute a statistic -- every value below is read straight out of the file.
// Where a list is sorted, that is presentation ordering, not derivation.

export interface NullBand {
  p5: number;
  p50: number;
  p95: number;
}

export interface PermutationNullSummary {
  nPermutations: number;
  observed: number;
  band: NullBand;
  pValue: number;
  /** The per-permutation AUROCs themselves. Present for exp1 (injected by
   * scripts/emit_results.py from results/exp1_permutation_null.json); empty for exp2,
   * whose raw array was never transcribed into the contract. */
  distribution: number[];
}

export function getPermutationNullSummary(id: "exp1" | "exp2"): PermutationNullSummary {
  const pn = getExperiment(id).permutation_null as unknown as {
    n_permutations: number;
    observed_auroc: number;
    null: { percentiles: Record<"5" | "50" | "95", number>; distribution?: number[] };
    p_value: number;
  };
  return {
    nPermutations: pn.n_permutations,
    observed: pn.observed_auroc,
    band: { p5: pn.null.percentiles["5"], p50: pn.null.percentiles["50"], p95: pn.null.percentiles["95"] },
    pValue: pn.p_value,
    distribution: pn.null.distribution ?? [],
  };
}

export interface Headline {
  value: number;
  lower: number;
  upper: number;
}

export function getHeadline(id: "exp1" | "exp2"): Headline {
  const h = getExperiment(id).headline as unknown as { value: number; ci95: { lower: number; upper: number } };
  return { value: h.value, lower: h.ci95.lower, upper: h.ci95.upper };
}

export function getExp1Caveat(): string {
  return (getExperiment("exp1").design as unknown as { caveat: string }).caveat;
}

export interface DyadScore {
  pairId: string;
  auroc: number;
}

/** exp2's twelve held-out dyad AUROCs, ranked high to low. Ranking is presentation only --
 * absolute scores read better sorted than in pair-id order, and no value is recomputed. */
export function getExp2PerDyadAuroc(): DyadScore[] {
  const scores = (getExperiment("exp2").per_dyad_scores as unknown as {
    scores: { auroc: Record<string, number> };
  }).scores.auroc;
  return Object.entries(scores)
    .map(([pairId, auroc]) => ({ pairId, auroc }))
    .sort((a, b) => b.auroc - a.auroc);
}

export function getExp2VsExp1Delta(): number {
  const c = getExperiment("exp2").comparison_to_exp1 as unknown as {
    per_family: Record<string, { delta: number }>;
  };
  return c.per_family.logistic_regression.delta;
}

export function getExp2DyadsAboveChance(): { above: number; total: number } {
  const c = getExperiment("exp2").comparison_to_exp1 as unknown as {
    n_dyads_above_chance: Record<string, number>;
  };
  return { above: c.n_dyads_above_chance.logistic_regression, total: getExperiment("exp2").per_dyad.length };
}

export function getExp3Exclusions(): Record<string, string> {
  return (getExperiment("exp3").exclusions ?? {}) as Record<string, string>;
}

export interface CurvePoint {
  k: number;
  sameMedian: number;
  sameSpread: [number, number];
  otherMedian: number;
  otherSpread: [number, number];
}

/** exp5's descriptive curve. `by_k`'s keys are strings in the JSON; sorted numerically
 * here so the curve draws left to right. */
export function getExp5Curve(): CurvePoint[] {
  const agg = (getExperiment("exp5") as unknown as {
    aggregate: {
      by_k: Record<string, {
        same_dyad_median: number;
        same_dyad_spread: [number, number];
        other_dyad_median: number;
        other_dyad_spread: [number, number];
      }>;
    };
  }).aggregate;
  return Object.entries(agg.by_k)
    .map(([k, v]) => ({
      k: Number(k),
      sameMedian: v.same_dyad_median,
      sameSpread: v.same_dyad_spread,
      otherMedian: v.other_dyad_median,
      otherSpread: v.other_dyad_spread,
    }))
    .sort((a, b) => a.k - b.k);
}

export function getExp5AggregateNote(): string {
  return (getExperiment("exp5") as unknown as { aggregate: { note: string } }).aggregate.note;
}

export interface EarlyLatePair {
  pairId: string;
  early: number;
  late: number;
  /** The observer is a DIFFERENT participant in each block -- roles switch. Any figure
   * showing early beside late has to say so. */
  earlyObserver: string;
  lateObserver: string;
}

/** exp6's per-dyad early and late decodability, in the same order as
 * tests.exp6_observer_early_late.result.dyad_ids. The `middle` bin exists in the data and
 * is deliberately not returned -- the test this screen reports is late minus early. */
export function getExp6EarlyLate(): EarlyLatePair[] {
  const perDyad = (getExperiment("exp6") as unknown as {
    per_dyad: {
      pair_id: string;
      observers: { early_block: string; late_block: string };
      positional_bins: { bin: string; auroc: number }[];
    }[];
  }).per_dyad;
  return perDyad.map((d) => {
    const byBin = Object.fromEntries(d.positional_bins.map((b) => [b.bin, b.auroc]));
    return {
      pairId: d.pair_id,
      early: byBin.early,
      late: byBin.late,
      earlyObserver: d.observers.early_block,
      lateObserver: d.observers.late_block,
    };
  });
}

/** S16's binding reporting constraint, verbatim. Never paraphrase this string. */
export function getExp6ReportingConstraint(): string {
  return (getExperiment("exp6") as unknown as {
    interpretation: { s16_reporting_constraint: string };
  }).interpretation.s16_reporting_constraint;
}

/** exp6's gate verdict is an object ({"1A": ..., "1B": ..., "2": ...}), unlike the other
 * experiments' plain string. */
export function getExp6GateVerdict(): Record<string, string> {
  return (getExperiment("exp6") as unknown as { gate_verdict: Record<string, string> }).gate_verdict;
}

export interface RankedInputSet {
  id: string;
  label: string;
  median: number;
  lower: number;
  upper: number;
  nAboveChance: number;
}

/** exp7's five input sets, ranked by median AUROC descending. Note `ci95` is centred on
 * the MEAN of the eleven per-fold AUROCs, not on the median plotted as the dot -- the
 * whisker and the dot are deliberately not concentric. */
export function getExp7Ranked(): RankedInputSet[] {
  const exp7 = getExperiment("exp7") as unknown as {
    input_set_labels: Record<string, string>;
    per_input_set: Record<string, {
      median_auroc: number;
      ci95: { lower: number; upper: number };
      n_dyads_above_chance: number;
    }>;
  };
  return Object.entries(exp7.per_input_set)
    .map(([id, v]) => ({
      id,
      label: exp7.input_set_labels[id],
      median: v.median_auroc,
      lower: v.ci95.lower,
      upper: v.ci95.upper,
      nAboveChance: v.n_dyads_above_chance,
    }))
    .sort((a, b) => b.median - a.median);
}

export function getExp7InterbrainVsDeceiver(): PairedTest {
  return (getExperiment("exp7") as unknown as {
    tests: Record<string, { result: PairedTest }>;
  }).tests.interbrain_vs_deceiver_eeg.result;
}

/** How many of exp7's comparisons carry a Benjamini-Hochberg-corrected p-value. This is
 * SIX, not ten: the four `*_vs_deceiver_eeg` comparisons are designated confirmatory /
 * secondary and are not part of the exploratory BH family. */
export function getExp7BhCorrectedCount(): number {
  return Object.keys(
    (getExperiment("exp7") as unknown as { tests_exploratory_bh: Record<string, unknown> })
      .tests_exploratory_bh,
  ).length;
}

export function getGeneratedAt(): string {
  return RESULTS.meta.generated_at;
}
