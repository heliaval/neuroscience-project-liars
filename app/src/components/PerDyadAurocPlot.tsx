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
  const width = 920;
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
      <div className="min-h-0 overflow-x-auto overflow-y-hidden lg:flex lg:flex-1 lg:flex-col lg:justify-end lg:[container-type:size]">
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
          const barHeight = 6;
          const xEnd = x(s.auroc);
          const barX = Math.min(chanceX, xEnd);
          const barWidth = Math.max(1, Math.abs(xEnd - chanceX));
          return (
            <g key={s.pairId}>
              <text x={marginLeft - 12} y={rowY + 4} textAnchor="end" fill="var(--color-ink-soft)">
                {s.pairId.replace(/_/g, " / ")}
              </text>
              {/* a faint full-width rail so the label reads as connected to the row
                  even when the bar itself starts at the chance line, well right of
                  where the label sits -- without it the gap reads as a mistake. */}
              <line
                x1={marginLeft}
                y1={rowY}
                x2={width - marginRight}
                y2={rowY}
                stroke="var(--color-hairline)"
                strokeWidth={1}
                opacity={0.4}
              />
              {/* a solid bar from the chance line to the score, not just a thin
                  connecting stroke -- grows from the chance line via scaleX so the
                  static width/x stay the true final geometry motion can rely on. */}
              <motion.rect
                x={barX}
                y={rowY - barHeight / 2}
                width={barWidth}
                height={barHeight}
                rx={barHeight / 2}
                fill="var(--color-signal-soft)"
                style={{ transformOrigin: `${chanceX}px ${rowY}px` }}
                initial={{ scaleX: 0 }}
                animate={{ scaleX: reveal ? 1 : 0 }}
                transition={{ duration: dur, ease: WT.ease, delay: reduced ? 0 : i * WT.stagger }}
              />
              <motion.circle
                cy={rowY}
                r={4.5}
                fill={s.auroc >= 0.5 ? "var(--color-ink)" : "var(--color-paper)"}
                stroke="var(--color-ink)"
                strokeWidth={1.5}
                initial={{ cx: chanceX, opacity: 0 }}
                animate={{ cx: reveal ? xEnd : chanceX, opacity: reveal ? 1 : 0 }}
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
