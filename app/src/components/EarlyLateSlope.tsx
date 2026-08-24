import { motion, useReducedMotion } from "motion/react";
import { WT } from "@/walkthrough/motion";
import type { EarlyLatePair } from "@/data/selectors";
import { chartBox } from "./chartBox";

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
/** Greedy label decluttering: pushes vertically-sorted labels apart until each
 * clears minGap from its neighbor, then re-centers the whole stack within
 * [top, bottom] if the push-down ran past the bottom. Dots/lines stay at each
 * pair's true value; only the text label position is adjusted, so a reader
 * comparing two nearby values still reads the true positions off the marks,
 * not the (sometimes slightly offset) label. */
function declutterY(
  items: { id: string; y: number }[],
  minGap: number,
  top: number,
  bottom: number,
): Map<string, number> {
  const sorted = [...items].sort((a, b) => a.y - b.y);
  for (let i = 1; i < sorted.length; i++) {
    if (sorted[i].y - sorted[i - 1].y < minGap) sorted[i].y = sorted[i - 1].y + minGap;
  }
  const overflow = sorted[sorted.length - 1].y - bottom;
  if (overflow > 0) {
    for (const s of sorted) s.y -= overflow;
    if (sorted[0].y < top) {
      sorted[0].y = top;
      for (let i = 1; i < sorted.length; i++) {
        if (sorted[i].y - sorted[i - 1].y < minGap) sorted[i].y = sorted[i - 1].y + minGap;
      }
    }
  }
  return new Map(sorted.map((s) => [s.id, s.y]));
}

/** Presentational only: the `sub` prefix repeats on all 20 marks and discriminates
 * nothing. The aria-label and the caption keep the full IDs, so nothing an assistive
 * technology reads is abbreviated. */
const shortId = (s: string) => s.replace(/^sub/, "");

export function EarlyLateSlope({
  pairs,
  reveal,
}: {
  pairs: EarlyLatePair[];
  reveal: boolean;
}) {
  const reduced = useReducedMotion();
  const width = 1080;
  const height = 460;
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

  const LABEL_MIN_GAP = 15;
  const earlyLabelY = declutterY(
    pairs.map((p) => ({ id: p.pairId, y: y(p.early) })),
    LABEL_MIN_GAP, marginTop, height - marginBottom,
  );
  const lateLabelY = declutterY(
    pairs.map((p) => ({ id: p.pairId, y: y(p.late) })),
    LABEL_MIN_GAP, marginTop, height - marginBottom,
  );

  const dur = reduced ? 0 : WT.base;
  const spring = reduced ? { duration: 0 } : WT.spring;

  return (
    <figure className="flex max-w-full flex-col lg:min-h-0 lg:flex-1">
      <div className="min-h-0 overflow-x-auto overflow-y-hidden lg:flex-1 lg:[container-type:size]">
      <svg
        viewBox={`0 0 ${width} ${height}`}
        role="img"
        aria-label={`Decodability in the early versus late block for each of ${pairs.length} pairs: ${pairs
          .map((p) => `${p.pairId} early ${p.early.toFixed(4)} late ${p.late.toFixed(4)}`)
          .join("; ")}`}
        className="block w-full min-w-[460px] font-mono text-[11px] lg:w-[var(--chart-fit-w)]"
        style={chartBox(width, height, true)}
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
        <text x={earlyX + 8} y={chanceY - 6} textAnchor="start" fill="var(--color-ink-faint)">
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
              {(() => {
                const labelY0 = earlyLabelY.get(p.pairId)!;
                return Math.abs(labelY0 - y0) > 2.5 ? (
                  <motion.line
                    x1={earlyX - 6} y1={y0} x2={earlyX - 6} y2={labelY0}
                    stroke="var(--color-hairline)" strokeWidth={1}
                    initial={{ opacity: 0 }} animate={{ opacity: reveal ? 1 : 0 }}
                    transition={{ duration: dur, ease: WT.ease, delay }}
                  />
                ) : null;
              })()}
              <motion.text
                x={earlyX - 10}
                y={earlyLabelY.get(p.pairId)! + 4}
                textAnchor="end"
                fontSize={9}
                fill="var(--color-ink-soft)"
                initial={{ opacity: 0 }}
                animate={{ opacity: reveal ? 1 : 0 }}
                transition={{ duration: dur, ease: WT.ease, delay }}
              >
                {shortId(p.earlyObserver)}
              </motion.text>
              {(() => {
                const labelY1 = lateLabelY.get(p.pairId)!;
                return Math.abs(labelY1 - y1) > 2.5 ? (
                  <motion.line
                    x1={lateX + 6} y1={y1} x2={lateX + 6} y2={labelY1}
                    stroke="var(--color-hairline)" strokeWidth={1}
                    initial={{ opacity: 0 }} animate={{ opacity: reveal ? 1 : 0 }}
                    transition={{ duration: dur, ease: WT.ease, delay: delay + (reduced ? 0 : WT.base) }}
                  />
                ) : null;
              })()}
              <motion.text
                x={lateX + 10}
                y={lateLabelY.get(p.pairId)! + 4}
                textAnchor="start"
                fontSize={9}
                fill="var(--color-ink-soft)"
                initial={{ opacity: 0 }}
                animate={{ opacity: reveal ? 1 : 0 }}
                transition={{ duration: dur, ease: WT.ease, delay: delay + (reduced ? 0 : WT.base) }}
              >
                {shortId(p.lateObserver)}
              </motion.text>
            </g>
          );
        })}

        <text x={width / 2} y={height - 10} textAnchor="middle" fill="var(--color-ink-faint)">
          decodability (AUROC) · one line per pair · labels are the observing participant (sub·) in each block
        </text>
      </svg>
      </div>
    </figure>
  );
}
