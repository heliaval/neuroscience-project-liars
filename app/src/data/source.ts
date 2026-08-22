// app/src/data/source.ts -- swapping to real results is a one-line change to this file.
// Today: reads the frozen fixture. Later: point this import at
// `../../../results/results.v1.json` once src/emit.py exists (plans/results-v1-fixture.md
// §51 step 10) and nothing else in the app changes.
import fixture from "../../../results/fixtures/results.v1.fixture.json";
import type { ResultsV1 } from "../types/results.v1";

export const RESULTS = fixture as unknown as ResultsV1;
