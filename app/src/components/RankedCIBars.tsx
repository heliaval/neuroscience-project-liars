import { motion, useReducedMotion } from "motion/react";
import { WT } from "@/walkthrough/motion";
import type { RankedInputSet } from "@/data/selectors";

/**
 * exp7's five input sets, ranked by median AUROC, each with its 95% CI as a whisker
 * against the 0.50 chance line. The whisker is computed over the eleven per-fold AUROCs
 * and is centred on their MEAN, not on the median plotted as the dot -- they are not
 * concentric, and that is correct, not a bug.
 */
export function RankedCIBars({ sets, reveal }: { sets: RankedInputSet[]; reveal: boolean }) {
  const reduced = useReducedMotion();
  const width = 640;
  const rowGap = 42;
  const marginLeft = 208;
  const marginRight = 24;
  const marginTop = 28;
  const height = marginTop + sets.length * rowGap + 32;
  const plotWidth = width - marginLeft - marginRight;

  const DOMAIN_MIN = 0.46;
  const DOMAIN_MAX = 0.56;
  const x = (v: number) => marginLeft + ((v - DOMAIN_MIN) / (DOMAIN_MAX - DOMAIN_MIN)) * plotWidth;
  const chanceX = x(0.5);
  const dur = reduced ? 0 : WT.base;

  return (
    <figure className="max-w-full overflow-x-auto">
      <svg
        viewBox={`0 0 ${width} ${height}`}
        role="img"
        aria-label={`Five input sets ranked by median AUROC: ${sets.map((s) => `${s.label} ${s.median.toFixed(4)}`).join("; ")}`}
        className="block h-[min(48vh,460px)] w-full min-w-[480px] font-mono text-[11px]"
      >
        <line
          x1={chanceX}
          y1={marginTop - 12}
          x2={chanceX}
          y2={height - 28}
          stroke="var(--color-hairline-strong)"
          strokeWidth={2}
        />
        <text x={chanceX} y={marginTop - 18} textAnchor="middle" fill="var(--color-ink-faint)">
          0.50 chance
        </text>

        {sets.map((s, i) => {
          const rowY = marginTop + 12 + i * rowGap;
          return (
            <g key={s.id}>
              <text x={marginLeft - 12} y={rowY + 4} textAnchor="end" fill="var(--color-ink-soft)">
                {s.label}
              </text>
              <text x={marginLeft - 12} y={rowY + 18} textAnchor="end" fill="var(--color-ink-faint)">
                {s.nAboveChance} of 11 above chance
              </text>

              <motion.line
                y1={rowY}
                y2={rowY}
                stroke="var(--color-hairline-strong)"
                strokeWidth={1}
                initial={{ x1: chanceX, x2: chanceX }}
                animate={{ x1: reveal ? x(s.lower) : chanceX, x2: reveal ? x(s.upper) : chanceX }}
                transition={{ duration: dur, ease: WT.ease, delay: reduced ? 0 : i * WT.stagger }}
              />
              {[s.lower, s.upper].map((v, j) => (
                <motion.line
                  key={j}
                  y1={rowY - 5}
                  y2={rowY + 5}
                  stroke="var(--color-hairline-strong)"
                  strokeWidth={1}
                  initial={{ x1: chanceX, x2: chanceX }}
                  animate={{ x1: reveal ? x(v) : chanceX, x2: reveal ? x(v) : chanceX }}
                  transition={{ duration: dur, ease: WT.ease, delay: reduced ? 0 : i * WT.stagger }}
                />
              ))}
              <motion.circle
                cy={rowY}
                r={5}
                fill="var(--color-ink)"
                initial={{ cx: chanceX, opacity: 0 }}
                animate={{ cx: reveal ? x(s.median) : chanceX, opacity: reveal ? 1 : 0 }}
                transition={{ duration: dur, ease: WT.ease, delay: reduced ? 0 : i * WT.stagger }}
              />
              <motion.text
                x={x(s.median)}
                y={rowY - 12}
                textAnchor="middle"
                fill="var(--color-ink)"
                initial={{ opacity: 0 }}
                animate={{ opacity: reveal ? 1 : 0 }}
                transition={{ duration: dur, ease: WT.ease, delay: reduced ? 0 : i * WT.stagger }}
              >
                {s.median.toFixed(4)}
              </motion.text>
            </g>
          );
        })}

        <text x={width - marginRight} y={height - 8} textAnchor="end" fill="var(--color-ink-faint)">
          median AUROC · whisker = 95% CI over 11 folds
        </text>
      </svg>
    </figure>
  );
}
