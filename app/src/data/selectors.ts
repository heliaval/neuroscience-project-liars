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
  "exp8_onset_lag",
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
