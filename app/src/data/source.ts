// app/src/data/source.ts -- the app's one point of contact with the results file.
// Reads the real emitted file, results/results.v1.json, which scripts/emit_results.py
// regenerates from the pipeline's real output. The fixture at
// results/fixtures/results.v1.fixture.json remains for reference and for the
// still-invented sections it is an input to, but the app no longer reads it directly.
import results from "../../../results/results.v1.json";
import type { ResultsV1 } from "../types/results.v1";

export const RESULTS = results as unknown as ResultsV1;
