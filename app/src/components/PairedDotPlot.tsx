import type { PairedTest } from "@/types/results.v1";

/**
 * General renderer for the project's one inferential procedure (§20): a
 * paired per-dyad delta against a zero line. This same shape later renders
 * exp3, exp5, exp6, and exp7's comparisons unchanged -- it is deliberately
 * NOT specific to exp4 (plans/web-app-scaffold.md §5.3).
 *
 * Must read equally well whether most dyads land above or below the line --
 * the real dyad-gain data has 8 of 10 below it, so "above" carries no
 * built-in positive framing here.
 */
export function PairedDotPlot({ test, unit = "Δ AUROC" }: { test: PairedTest; unit?: string }) {
  const width = 640;
  const height = 96 + test.n * 30;
  const marginLeft = 96;
  const marginRight = 32;
  const plotWidth = width - marginLeft - marginRight;

  const domainMax = Math.max(0.06, ...test.deltas.map((d) => Math.abs(d))) * 1.15;
  const xScale = (v: number) => marginLeft + ((v + domainMax) / (2 * domainMax)) * plotWidth;
  const zeroX = xScale(0);

  const rows = test.dyad_ids.map((id, i) => ({ id, delta: test.deltas[i], y: 40 + i * 30 }));

  return (
    <figure className="my-8 max-w-full overflow-x-auto">
      <svg
        viewBox={`0 0 ${width} ${height}`}
        role="img"
        aria-label={`Per-dyad ${unit} for ${test.n} dyads, median ${test.median_delta.toFixed(4)}`}
        className="block w-full min-w-[420px] font-mono text-[11px]"
      >
        {/* zero line */}
        <line x1={zeroX} y1={20} x2={zeroX} y2={height - 24} stroke="var(--color-hairline-strong)" strokeWidth={1} />
        <text x={zeroX} y={14} textAnchor="middle" fill="var(--color-ink-faint)">
          0
        </text>

        {/* median marker */}
        <line
          x1={xScale(test.median_delta)}
          y1={20}
          x2={xScale(test.median_delta)}
          y2={height - 24}
          stroke="var(--color-signal)"
          strokeWidth={1.5}
          strokeDasharray="3 3"
        />

        {rows.map((r) => (
          <g key={r.id}>
            <text x={marginLeft - 10} y={r.y + 4} textAnchor="end" fill="var(--color-ink-soft)">
              {r.id.replace(/_/g, " / ")}
            </text>
            <line x1={zeroX} y1={r.y} x2={xScale(r.delta)} y2={r.y} stroke="var(--color-hairline)" strokeWidth={1} />
            <circle
              cx={xScale(r.delta)}
              cy={r.y}
              r={5}
              fill={r.delta >= 0 ? "var(--color-ink)" : "var(--color-paper)"}
              stroke="var(--color-ink)"
              strokeWidth={1.5}
            />
          </g>
        ))}

        <text x={width - marginRight} y={height - 6} textAnchor="end" fill="var(--color-ink-faint)">
          {unit}
        </text>
      </svg>

      <figcaption className="mt-4 grid grid-cols-2 gap-x-6 gap-y-1 font-mono text-[12px] text-ink-soft sm:grid-cols-4">
        <span>median {test.median_delta.toFixed(4)}</span>
        <span>
          {test.n_positive} of {test.n} above zero
        </span>
        <span>sign p {test.sign_test_p.toFixed(3)}</span>
        <span>permutation p {test.permutation_p.toFixed(3)}</span>
        <span className="col-span-2 sm:col-span-4">
          95% CI [{test.ci95.lower?.toFixed(4)}, {test.ci95.upper?.toFixed(4)}]
        </span>
      </figcaption>
    </figure>
  );
}
