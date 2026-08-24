import { motion, useReducedMotion } from "motion/react";
import { WT } from "@/walkthrough/motion";
import type { DyadScore } from "@/data/selectors";
import { chartBox } from "./chartBox";

/**
 * One dot per held-out dyad, absolute AUROC against the 0.50 chance line, with the
 * pooled result as a dashed reference. Fixed [0.44, 0.58] domain rather than auto-scaled:
 * every score sits within a few points of chance, and auto-scaling would stretch a ~0.06
 * total spread across the full width and manufacture separation. Same reasoning as
 * ConditionComparison's fixed [0.40, 0.60].
 *
 * Rows draw outward from the chance line, staggered by rank.
 */
export function PerDyadAurocPlot({
  scores,
  reference,
  referenceLabel,
  reveal,
}: {
  scores: DyadScore[];
  reference: number;
  referenceLabel: string;
  reveal: boolean;
}) {
  const reduced = useReducedMotion();
  const width = 640;
  const rowGap = 26;
  const marginLeft = 104;
  const marginRight = 28;
  const marginTop = 34;
  const height = marginTop + scores.length * rowGap + 34;
  const plotWidth = width - marginLeft - marginRight;

  const DOMAIN_MIN = 0.44;
  const DOMAIN_MAX = 0.58;
  const x = (v: number) => marginLeft + ((v - DOMAIN_MIN) / (DOMAIN_MAX - DOMAIN_MIN)) * plotWidth;
  const chanceX = x(0.5);
  const dur = reduced ? 0 : WT.base;

  return (
    <figure className="flex max-w-full flex-col lg:min-h-0 lg:flex-1">
      <div className="min-h-0 overflow-x-auto lg:flex-1 lg:[container-type:size]">
      <svg
        viewBox={`0 0 ${width} ${height}`}
        role="img"
        aria-label={`AUROC for each of ${scores.length} held-out dyads against a 0.50 chance line, with the pooled value ${reference.toFixed(4)} marked.`}
        className="block w-full min-w-[440px] font-mono text-[11px] lg:w-[var(--chart-fit-w)]"
        style={chartBox(width, height, true)}
      >
        <line
          x1={chanceX}
          y1={marginTop - 14}
          x2={chanceX}
          y2={height - 30}
          stroke="var(--color-hairline-strong)"
          strokeWidth={2}
        />
        <text x={chanceX} y={marginTop - 20} textAnchor="middle" fill="var(--color-ink-faint)">
          0.50 chance
        </text>

        <line
          x1={x(reference)}
          y1={marginTop - 14}
          x2={x(reference)}
          y2={height - 30}
          stroke="var(--color-signal)"
          strokeWidth={1.5}
          strokeDasharray="3 3"
        />

        {scores.map((s, i) => {
          const rowY = marginTop + 8 + i * rowGap;
          return (
            <g key={s.pairId}>
              <text x={marginLeft - 12} y={rowY + 4} textAnchor="end" fill="var(--color-ink-soft)">
                {s.pairId.replace(/_/g, " / ")}
              </text>
              <motion.line
                y1={rowY}
                y2={rowY}
                x1={chanceX}
                stroke="var(--color-hairline)"
                strokeWidth={1}
                initial={{ x2: chanceX }}
                animate={{ x2: reveal ? x(s.auroc) : chanceX }}
                transition={{ duration: dur, ease: WT.ease, delay: reduced ? 0 : i * WT.stagger }}
              />
              <motion.circle
                cy={rowY}
                r={5}
                fill={s.auroc >= 0.5 ? "var(--color-ink)" : "var(--color-paper)"}
                stroke="var(--color-ink)"
                strokeWidth={1.5}
                initial={{ cx: chanceX, opacity: 0 }}
                animate={{ cx: reveal ? x(s.auroc) : chanceX, opacity: reveal ? 1 : 0 }}
                transition={{ duration: dur, ease: WT.ease, delay: reduced ? 0 : i * WT.stagger }}
              />
            </g>
          );
        })}

        <text x={width - marginRight} y={height - 8} textAnchor="end" fill="var(--color-ink-faint)">
          AUROC · dashed rule = {referenceLabel} {reference.toFixed(4)}
        </text>
      </svg>
      </div>
    </figure>
  );
}
