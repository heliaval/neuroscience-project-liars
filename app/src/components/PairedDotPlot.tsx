import type { PairedTest } from "@/types/results.v1";
import { motion, useReducedMotion } from "motion/react";
import { WT } from "@/walkthrough/motion";
import { chartBox } from "./chartBox";

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
/** `reveal` is opt-in and exists for the walkthrough only. When it is omitted -- which is
 * every dashboard call site -- rendering is exactly what it always was: rows drawn at rest,
 * no motion. */
export function PairedDotPlot({
  test,
  unit = "Δ AUROC",
  reveal,
  fit,
}: {
  test: PairedTest;
  unit?: string;
  reveal?: boolean;
  /** Opt-in, walkthrough only; every dashboard call site omits this. */
  fit?: boolean;
}) {
  const reduced = useReducedMotion();
  const animated = reveal !== undefined;
  const shown = !animated || reveal === true;
  const width = 880;
  const height = 96 + test.n * 30;
  const marginLeft = 96;
  const marginRight = 32;
  const plotWidth = width - marginLeft - marginRight;

  const domainMax = Math.max(0.06, ...test.deltas.map((d) => Math.abs(d))) * 1.15;
  const xScale = (v: number) => marginLeft + ((v + domainMax) / (2 * domainMax)) * plotWidth;
  const zeroX = xScale(0);

  const rows = test.dyad_ids.map((id, i) => ({ id, delta: test.deltas[i], y: 40 + i * 30 }));

  const svg = (
    <svg
      viewBox={`0 0 ${width} ${height}`}
      role="img"
      aria-label={`Per-dyad ${unit} for ${test.n} dyads, median ${test.median_delta.toFixed(4)}`}
      className={`block w-full min-w-[420px] font-mono text-[11px]${fit ? " lg:w-[var(--chart-fit-w)]" : ""}`}
      style={chartBox(width, height, fit)}
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

        {rows.map((r, i) => (
          <g key={r.id}>
            <text x={marginLeft - 10} y={r.y + 4} textAnchor="end" fill="var(--color-ink-soft)">
              {r.id.replace(/_/g, " / ")}
            </text>
            <motion.line
              y1={r.y}
              y2={r.y}
              x1={zeroX}
              stroke="var(--color-hairline)"
              strokeWidth={1}
              initial={animated ? { x2: zeroX } : false}
              animate={{ x2: shown ? xScale(r.delta) : zeroX }}
              transition={{
                duration: reduced || !animated ? 0 : WT.base,
                ease: WT.ease,
                delay: reduced || !animated ? 0 : i * WT.stagger,
              }}
            />
            <motion.circle
              cy={r.y}
              r={5}
              fill={r.delta >= 0 ? "var(--color-ink)" : "var(--color-paper)"}
              stroke="var(--color-ink)"
              strokeWidth={1.5}
              initial={animated ? { cx: zeroX, opacity: 0 } : false}
              animate={{ cx: shown ? xScale(r.delta) : zeroX, opacity: shown ? 1 : 0 }}
              transition={{
                duration: reduced || !animated ? 0 : WT.base,
                ease: WT.ease,
                delay: reduced || !animated ? 0 : i * WT.stagger,
              }}
            />
          </g>
        ))}

        <text x={width - marginRight} y={height - 6} textAnchor="end" fill="var(--color-ink-faint)">
          {unit}
        </text>
    </svg>
  );

  return (
    <figure className={fit ? "flex max-w-full flex-col lg:min-h-0 lg:flex-1" : "my-8 max-w-full overflow-x-auto"}>
      {fit ? <div className="min-h-0 overflow-x-auto overflow-y-hidden lg:flex-1 lg:[container-type:size]">{svg}</div> : svg}

      <figcaption className="mt-4 grid shrink-0 grid-cols-2 gap-x-6 gap-y-1 font-mono text-[12px] text-ink-soft sm:grid-cols-4">
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
