import { motion, useReducedMotion } from "motion/react";
import { WT } from "@/walkthrough/motion";
import type { EarlyLatePair } from "@/data/selectors";

/**
 * exp6's per-dyad Early -> Late comparison: two real measured points per pair, joined.
 * This is the comparison the test actually consumes (late minus early); the `middle` bin
 * in the data is deliberately not drawn.
 *
 * The observer is a DIFFERENT participant in each block -- roles switch between the early
 * and late halves -- so the row label names both. The caller's caveat says it in words.
 *
 * Wording discipline (S16): this figure shows statistical decodability of a signal to a
 * fitted model. It does not show, and must never be captioned as showing, the observer
 * noticing, sensing, or knowing anything.
 *
 * Hollow vs filled endpoint marks distinguish below- from above-chance, exactly as
 * PairedDotPlot does. No color encodes direction.
 */
export function EarlyLateSlope({ pairs, reveal }: { pairs: EarlyLatePair[]; reveal: boolean }) {
  const reduced = useReducedMotion();
  const width = 640;
  const height = 340;
  const marginLeft = 116;
  const marginRight = 116;
  const marginTop = 44;
  const marginBottom = 44;
  const plotHeight = height - marginTop - marginBottom;

  const DOMAIN_MIN = 0.42;
  const DOMAIN_MAX = 0.60;
  const earlyX = marginLeft;
  const lateX = width - marginRight;
  const y = (v: number) => marginTop + plotHeight - ((v - DOMAIN_MIN) / (DOMAIN_MAX - DOMAIN_MIN)) * plotHeight;
  const chanceY = y(0.5);

  const dur = reduced ? 0 : WT.base;
  const spring = reduced ? { duration: 0 } : WT.spring;

  return (
    <figure className="max-w-full overflow-x-auto">
      <svg
        viewBox={`0 0 ${width} ${height}`}
        role="img"
        aria-label={`Decodability in the early versus late block for each of ${pairs.length} pairs: ${pairs
          .map((p) => `${p.pairId} early ${p.early.toFixed(4)} late ${p.late.toFixed(4)}`)
          .join("; ")}`}
        className="block h-[min(26vh,260px)] w-full min-w-[460px] font-mono text-[11px]"
      >
        {/* both axes present before the reveal -- the slopes draw between them */}
        <line x1={earlyX} y1={marginTop - 10} x2={earlyX} y2={height - marginBottom + 10} stroke="var(--color-hairline)" />
        <line x1={lateX} y1={marginTop - 10} x2={lateX} y2={height - marginBottom + 10} stroke="var(--color-hairline)" />
        <text x={earlyX} y={marginTop - 18} textAnchor="middle" fill="var(--color-ink-faint)">
          EARLY BLOCK
        </text>
        <text x={lateX} y={marginTop - 18} textAnchor="middle" fill="var(--color-ink-faint)">
          LATE BLOCK
        </text>

        <line
          x1={earlyX}
          y1={chanceY}
          x2={lateX}
          y2={chanceY}
          stroke="var(--color-hairline-strong)"
          strokeWidth={2}
        />
        <text x={earlyX - 10} y={chanceY + 4} textAnchor="end" fill="var(--color-ink-faint)">
          0.50
        </text>

        {pairs.map((p, i) => {
          const delay = reduced ? 0 : i * WT.stagger * 1.6;
          const y0 = y(p.early);
          const y1 = y(p.late);
          return (
            <g key={p.pairId}>
              <motion.line
                x1={earlyX}
                y1={y0}
                y2={y1}
                stroke="var(--color-ink-soft)"
                strokeWidth={1.25}
                initial={{ x2: earlyX, opacity: 0 }}
                animate={{ x2: reveal ? lateX : earlyX, opacity: reveal ? 1 : 0 }}
                transition={{ duration: reduced ? 0 : WT.slow, ease: WT.ease, delay }}
              />
              <motion.circle
                cx={earlyX}
                cy={y0}
                r={4.5}
                fill={p.early >= 0.5 ? "var(--color-ink)" : "var(--color-paper)"}
                stroke="var(--color-ink)"
                strokeWidth={1.5}
                initial={{ scale: 0, opacity: 0 }}
                animate={{ scale: reveal ? 1 : 0, opacity: reveal ? 1 : 0 }}
                transition={{ ...spring, delay }}
                style={{ transformOrigin: `${earlyX}px ${y0}px` }}
              />
              <motion.circle
                cx={lateX}
                cy={y1}
                r={4.5}
                fill={p.late >= 0.5 ? "var(--color-ink)" : "var(--color-paper)"}
                stroke="var(--color-ink)"
                strokeWidth={1.5}
                initial={{ scale: 0, opacity: 0 }}
                animate={{ scale: reveal ? 1 : 0, opacity: reveal ? 1 : 0 }}
                transition={{ ...spring, delay: delay + (reduced ? 0 : WT.base) }}
                style={{ transformOrigin: `${lateX}px ${y1}px` }}
              />
              <motion.text
                x={earlyX - 10}
                y={y0 + 4}
                textAnchor="end"
                fill="var(--color-ink-soft)"
                initial={{ opacity: 0 }}
                animate={{ opacity: reveal ? 1 : 0 }}
                transition={{ duration: dur, ease: WT.ease, delay }}
              >
                {p.earlyObserver}
              </motion.text>
              <motion.text
                x={lateX + 10}
                y={y1 + 4}
                textAnchor="start"
                fill="var(--color-ink-soft)"
                initial={{ opacity: 0 }}
                animate={{ opacity: reveal ? 1 : 0 }}
                transition={{ duration: dur, ease: WT.ease, delay: delay + (reduced ? 0 : WT.base) }}
              >
                {p.lateObserver}
              </motion.text>
            </g>
          );
        })}

        <text x={width / 2} y={height - 10} textAnchor="middle" fill="var(--color-ink-faint)">
          decodability (AUROC) · one line per pair · labels name the observing participant in each block
        </text>
      </svg>
    </figure>
  );
}
