# Walkthrough app

This is the judge-facing walkthrough for [Can Your Brain Learn a Liar?](../README.md). It presents experiments 1 through 7 as a guided sequence, one screen per experiment plus an opening and a closing screen.

The app performs no inference. It reads `results/results.v1.json` through `src/data/source.ts`, the app's single point of contact with the results file, and renders precomputed numbers. There is no model running client-side, and no live computation of any kind.

## Where the numbers come from

`src/data/source.ts` imports `results/results.v1.json` directly and casts it to the `ResultsV1` type. That type lives at `src/types/results.v1.d.ts` and is generated from `results/schema/results.v1.schema.json`, not written by hand. If the schema changes, regenerate the type with `npm run gen:types` before touching any component that reads new fields.

## Scripts

- `npm run dev` starts the Vite dev server.
- `npm run build` type-checks with `tsc -b` and produces a production build in `dist/`.
- `npm run lint` runs oxlint.
- `npm run preview` serves the production build locally.
- `npm run gen:types` regenerates `src/types/results.v1.d.ts` from `results/schema/results.v1.schema.json`.
- `npm run validate:fixture` checks `results/fixtures/results.v1.fixture.json` against the schema.
- `npm run validate:results` checks the real `results/results.v1.json` against the schema. This is the same check `scripts/validate_results.py` runs from the Python side.

For how the results file itself gets produced, see the root [README](../README.md#how-the-results-were-produced).
