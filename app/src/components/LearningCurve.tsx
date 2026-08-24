import { motion, useReducedMotion } from "motion/react";
import { WT } from "@/walkthrough/motion";
import type { CurvePoint } from "@/data/selectors";
import { chartBox } from "./chartBox";

/**
 * exp5's descriptive curve: median AUROC against training-set size k, for models trained
 * on the pair's own history versus other pairs'. Fixed [0.42, 0.58] domain for the same
 * reason ConditionComparison fixes its own -- everything sits near chance and auto-scaling
 * would manufacture separation.
 *
 * This figure is NOT the test. The caller says so; so does the JSON, in aggregate.note.
 */
export function LearningCurve({ points, reveal }: { points: CurvePoint[]; reveal: boolean }) {
  const reduced = useReducedMotion();
  const width = 640;
  const height = 300;
  const marginLeft = 52;
  const marginRight = 24;
  const marginTop = 24;
  const marginBottom = 44;
  const plotWidth = width - marginLeft - marginRight;
  const plotHeight = height - marginTop - marginBottom;

  const DOMAIN_MIN = 0.42;
  const DOMAIN_MAX = 0.58;
  const ks = points.map((p) => p.k);
  const kMin = Math.min(...ks);
  const kMax = Math.max(...ks);

  const x = (k: number) => marginLeft + ((k - kMin) / (kMax - kMin)) * plotWidth;
  const y = (v: number) => marginTop + plotHeight - ((v - DOMAIN_MIN) / (DOMAIN_MAX - DOMAIN_MIN)) * plotHeight;
  const chanceY = y(0.5);
  const dur = reduced ? 0 : WT.slow;

  const line = (get: (p: CurvePoint) => number) =>
    points.map((p, i) => `${i === 0 ? "M" : "L"} ${x(p.k)} ${y(get(p))}`).join(" ");
  const band = (lo: (p: CurvePoint) => number, hi: (p: CurvePoint) => number) =>
    [
      ...points.map((p, i) => `${i === 0 ? "M" : "L"} ${x(p.k)} ${y(hi(p))}`),
      ...[...points].reverse().map((p) => `L ${x(p.k)} ${y(lo(p))}`),
      "Z",
    ].join(" ");

  return (
    <figure className="flex max-w-full flex-col lg:min-h-0 lg:flex-1">
      <div className="min-h-0 overflow-x-auto overflow-y-hidden lg:flex-1 lg:[container-type:size]">
      <svg
        viewBox={`0 0 ${width} ${height}`}
        role="img"
        aria-label={`Median AUROC against training set size, for same-dyad and other-dyad training data, at k = ${ks.join(", ")}. Descriptive only; this figure is not the statistical test.`}
        className="block w-full min-w-[440px] font-mono text-[11px] lg:w-[var(--chart-fit-w)]"
        style={chartBox(width, height, true)}
      >
        <line x1={marginLeft} y1={marginTop} x2={marginLeft} y2={marginTop + plotHeight} stroke="var(--color-hairline)" />
        <line
          x1={marginLeft}
          y1={chanceY}
          x2={width - marginRight}
          y2={chanceY}
          stroke="var(--color-hairline-strong)"
          strokeWidth={2}
        />
        <text x={marginLeft - 8} y={chanceY + 4} textAnchor="end" fill="var(--color-ink-faint)">
          0.50
        </text>
        <text x={marginLeft - 8} y={y(DOMAIN_MAX) + 4} textAnchor="end" fill="var(--color-ink-faint)">
          {DOMAIN_MAX.toFixed(2)}
        </text>
        <text x={marginLeft - 8} y={y(DOMAIN_MIN) + 4} textAnchor="end" fill="var(--color-ink-faint)">
          {DOMAIN_MIN.toFixed(2)}
        </text>

        {points.map((p) => (
          <text key={p.k} x={x(p.k)} y={height - 22} textAnchor="middle" fill="var(--color-ink-faint)">
            {p.k}
          </text>
        ))}
        <text x={width - marginRight} y={height - 6} textAnchor="end" fill="var(--color-ink-faint)">
          training trials (k)
        </text>

        <motion.g initial={{ opacity: 0 }} animate={{ opacity: reveal ? 1 : 0 }} transition={{ duration: dur * 0.5, ease: WT.ease }}>
          <path d={band((p) => p.otherSpread[0], (p) => p.otherSpread[1])} fill="var(--color-hairline)" opacity={0.35} />
          <path d={band((p) => p.sameSpread[0], (p) => p.sameSpread[1])} fill="var(--color-signal-soft)" opacity={0.55} />
        </motion.g>

        <motion.path
          d={line((p) => p.otherMedian)}
          fill="none"
          stroke="var(--color-ink-faint)"
          strokeWidth={1.5}
          strokeDasharray="4 3"
          initial={{ pathLength: 0 }}
          animate={{ pathLength: reveal ? 1 : 0 }}
          transition={{ duration: dur, ease: WT.ease }}
        />
        <motion.path
          d={line((p) => p.sameMedian)}
          fill="none"
          stroke="var(--color-ink)"
          strokeWidth={2}
          initial={{ pathLength: 0 }}
          animate={{ pathLength: reveal ? 1 : 0 }}
          transition={{ duration: dur, ease: WT.ease }}
        />

        {/* a backing plate behind the legend -- it sits right where the confidence
            bands are tallest, and without one the text reads directly against the
            shaded fill underneath it. */}
        <rect
          x={width - marginRight - 158}
          y={marginTop - 8}
          width={158}
          height={30}
          fill="var(--color-paper)"
          opacity={0.82}
        />
        <text x={width - marginRight} y={marginTop + 4} textAnchor="end" fill="var(--color-ink)">
          same pair's history
        </text>
        <text x={width - marginRight} y={marginTop + 20} textAnchor="end" fill="var(--color-ink-faint)">
          other pairs' history
        </text>
      </svg>
      </div>
    </figure>
  );
}
