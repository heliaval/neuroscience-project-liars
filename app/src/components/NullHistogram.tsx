import { useMemo } from "react";
import { motion, useReducedMotion } from "motion/react";
import { WT } from "@/walkthrough/motion";
import { chartBox } from "./chartBox";

/**
 * The permutation null as it actually came out: one bar per bin over all
 * `nPermutations` shuffled-label AUROCs, with the observed value as a tick against it.
 * The bars are the real distribution -- binning is presentation of values already in
 * results.v1.json, and every statistic in the caption (percentiles, p) is read from the
 * file, not recomputed here.
 *
 * Same no-red/no-green discipline as PairedDotPlot: nothing encodes good or bad. The
 * histogram is what chance looks like; the tick is what happened; the reader compares.
 */
const BIN_COUNT = 34;
const DOMAIN_MIN = 0.46;
const DOMAIN_MAX = 0.56;

export function NullHistogram({
  distribution,
  observed,
  band,
  pValue,
  nPermutations,
  reveal,
}: {
  distribution: number[];
  observed: number;
  band: { p5: number; p50: number; p95: number };
  pValue: number;
  nPermutations: number;
  reveal: boolean;
}) {
  const reduced = useReducedMotion();
  const width = 640;
  const height = 320;
  const marginLeft = 28;
  const marginRight = 28;
  const marginTop = 40;
  const baselineY = height - 24;
  const plotWidth = width - marginLeft - marginRight;
  const plotHeight = baselineY - marginTop;

  const x = (v: number) => marginLeft + ((v - DOMAIN_MIN) / (DOMAIN_MAX - DOMAIN_MIN)) * plotWidth;
  const binWidth = plotWidth / BIN_COUNT;

  const bins = useMemo(() => {
    const counts = new Array<number>(BIN_COUNT).fill(0);
    for (const v of distribution) {
      const t = (v - DOMAIN_MIN) / (DOMAIN_MAX - DOMAIN_MIN);
      const i = Math.min(BIN_COUNT - 1, Math.max(0, Math.floor(t * BIN_COUNT)));
      counts[i] += 1;
    }
    return counts;
  }, [distribution]);

  const maxCount = Math.max(1, ...bins);
  const dur = reduced ? 0 : WT.base;

  return (
    <figure className="flex max-w-full flex-col lg:min-h-0 lg:flex-1">
      <div className="min-h-0 overflow-x-auto overflow-y-hidden lg:flex lg:flex-1 lg:flex-col lg:justify-end lg:[container-type:size]">
      <svg
        viewBox={`0 0 ${width} ${height}`}
        role="img"
        aria-label={`Histogram of ${nPermutations} shuffled-label AUROCs. The 5th to 95th percentile spans ${band.p5.toFixed(4)} to ${band.p95.toFixed(4)}; the observed value is ${observed.toFixed(4)}, permutation p ${pValue.toFixed(4)}.`}
        className="block w-full min-w-[440px] font-mono text-[11px] lg:w-[var(--chart-fit-w)]"
        style={chartBox(width, height, true)}
      >
        {/* the baseline is present before the reveal -- bars grow up out of it */}
        <line
          x1={marginLeft}
          y1={baselineY}
          x2={width - marginRight}
          y2={baselineY}
          stroke="var(--color-hairline-strong)"
          strokeWidth={1}
        />
        <text x={marginLeft} y={baselineY + 18} textAnchor="middle" fill="var(--color-ink-faint)">
          {DOMAIN_MIN.toFixed(2)}
        </text>
        <text x={width - marginRight} y={baselineY + 18} textAnchor="middle" fill="var(--color-ink-faint)">
          {DOMAIN_MAX.toFixed(2)}
        </text>

        {bins.map((count, i) => {
          const h = (count / maxCount) * plotHeight;
          const bx = marginLeft + i * binWidth;
          return (
            <motion.rect
              key={i}
              x={bx + 0.5}
              y={baselineY - h}
              width={Math.max(1, binWidth - 1)}
              height={h}
              fill="var(--color-signal-soft)"
              /* motion's initial/animate on SVG `height`+`y` together left the attribute
               * unset ("Expected length, undefined" console error) on the installed motion
               * version -- height/y are set as static final-geometry props instead, and the
               * reveal is driven by a bottom-anchored scaleY transform, which motion animates
               * reliably as a CSS transform. */
              style={{ transformOrigin: `${bx + binWidth / 2}px ${baselineY}px` }}
              initial={{ scaleY: 0 }}
              animate={{ scaleY: reveal ? 1 : 0 }}
              transition={{
                duration: dur,
                ease: WT.ease,
                delay: reduced ? 0 : i * (WT.staggerTight * 4),
              }}
            />
          );
        })}

        {/* percentile marks, read straight from the file */}
        {([["5th", band.p5], ["95th", band.p95]] as const).map(([label, v]) => (
          <motion.g
            key={label}
            initial={{ opacity: 0 }}
            animate={{ opacity: reveal ? 1 : 0 }}
            transition={{ duration: dur, ease: WT.ease, delay: reduced ? 0 : WT.base * 0.7 }}
          >
            <line
              x1={x(v)}
              y1={marginTop - 6}
              x2={x(v)}
              y2={baselineY}
              stroke="var(--color-hairline-strong)"
              strokeWidth={1}
              strokeDasharray="2 3"
            />
            <text x={x(v)} y={marginTop - 12} textAnchor="middle" fill="var(--color-ink-faint)">
              {label}
            </text>
          </motion.g>
        ))}

        <motion.g
          initial={{ opacity: 0 }}
          animate={{ opacity: reveal ? 1 : 0 }}
          transition={{ duration: dur, ease: WT.ease, delay: reduced ? 0 : WT.base }}
        >
          <line
            x1={x(observed)}
            y1={marginTop - 18}
            x2={x(observed)}
            y2={baselineY + 8}
            stroke="var(--color-ink)"
            strokeWidth={2}
          />
          <text x={x(observed)} y={marginTop - 24} textAnchor="middle" fill="var(--color-ink)">
            observed {observed.toFixed(4)}
          </text>
        </motion.g>
      </svg>
      </div>

      <figcaption className="mt-3 shrink-0 font-mono text-[12px] text-ink-soft">
        {nPermutations} label shuffles, every one of them plotted. 5th-95th percentile{" "}
        {band.p5.toFixed(4)}-{band.p95.toFixed(4)}, median {band.p50.toFixed(4)}. Permutation p{" "}
        {pValue.toFixed(4)}.
      </figcaption>
    </figure>
  );
}
