import { useEffect, useState } from "react";
import { EXP4_CONDITIONS, getConditionScores, getConditionSummary, type Exp4ConditionId } from "@/data/selectors";

const CONDITION_LABEL: Record<Exp4ConditionId, string> = {
  universal: "STRANGER",
  person_specific: "PERSON-SPECIFIC",
  dyad_specific: "DYAD-SPECIFIC",
};

// Axis domain: presentation geometry, not data (plans/frontend-stranger-vs-familiar.md §5.1).
const DOMAIN_MIN = 0.4;
const DOMAIN_MAX = 0.6;
const CHANCE = 0.5;

/**
 * Levels figure for §32 "Stranger vs Familiar": absolute AUROC per condition,
 * against the 0.50 chance line, for all ten dyads at once.
 *
 * The axis domain is FIXED at [0.40, 0.60], centered on chance, and does NOT
 * auto-scale to the data. This is the single most important decision in this
 * figure. The observed values span only 0.4582-0.5768; auto-scaling to that
 * narrow range would stretch a ~0.02 median gap across a quarter of the
 * figure's width and make three near-chance models look dramatically
 * separated. A fixed +/-0.10 window around chance keeps the true reading --
 * all three conditions sit close to chance, and the gaps between them are
 * small -- as the figure's immediate visual impression.
 *
 * No fill, stroke, or token in this figure encodes good or bad. The headline
 * finding is a null (the dyad-specific condition, the one this project's
 * hypothesis predicted would win, scores lowest of the three), and the figure
 * must read exactly as well with any of the three conditions "winning."
 * Same no-red/green discipline as PairedDotPlot.
 */
export function ConditionComparison({
  selected,
  maxHeight,
}: {
  selected: Exp4ConditionId;
  /** Opt-in CSS height (e.g. "min(34vh,340px)"), for the walkthrough only -- the
   * dashboard call site omits this, so the chart keeps scaling purely by width
   * exactly as it always has. */
  maxHeight?: string;
}) {
  const width = 640;
  const anchor = getConditionScores("universal");
  const rows = anchor.map((row, i) => {
    const byCondition = Object.fromEntries(
      EXP4_CONDITIONS.map((c) => [c, getConditionScores(c)[i].auroc]),
    ) as Record<Exp4ConditionId, number>;
    return { pair_id: row.pair_id, byCondition };
  });
  const marginLeft = 96;
  const marginRight = 32;
  const plotTop = 32;
  const rowGap = 32;
  const height = plotTop + rows.length * rowGap + 40;
  const plotWidth = width - marginLeft - marginRight;

  const xScale = (v: number) => marginLeft + ((v - DOMAIN_MIN) / (DOMAIN_MAX - DOMAIN_MIN)) * plotWidth;
  const chanceX = xScale(CHANCE);
  const summary = getConditionSummary(selected);

  const [reducedMotion, setReducedMotion] = useState(
    () => window.matchMedia("(prefers-reduced-motion: reduce)").matches,
  );
  useEffect(() => {
    const mq = window.matchMedia("(prefers-reduced-motion: reduce)");
    const onChange = () => setReducedMotion(mq.matches);
    mq.addEventListener("change", onChange);
    return () => mq.removeEventListener("change", onChange);
  }, []);

  return (
    <figure className="my-8 max-w-full overflow-x-auto">
      <svg
        viewBox={`0 0 ${width} ${height}`}
        role="img"
        aria-label={`AUROC by training condition for 10 dyads, fixed axis 0.40 to 0.60. ${CONDITION_LABEL[selected]} selected: median ${summary.median.toFixed(4)}, ${summary.nAboveChance} of ${summary.n} dyads above chance.`}
        className={`block min-w-[440px] font-mono text-[11px] ${maxHeight ? "max-w-full" : "w-full"}`}
        style={maxHeight ? { aspectRatio: `${width} / ${height}`, maxHeight } : undefined}
      >
        <text x={marginLeft} y={16} textAnchor="middle" fill="var(--color-ink-faint)">
          0.40
        </text>
        <text x={width - marginRight} y={16} textAnchor="middle" fill="var(--color-ink-faint)">
          0.60
        </text>

        {/* chance line -- the heaviest rule. This is a measured baseline, not a drawing
            convention: the majority-class model scores exactly 0.50 for every dyad. */}
        <line
          x1={chanceX}
          y1={plotTop - 4}
          x2={chanceX}
          y2={height - 32}
          stroke="var(--color-hairline-strong)"
          strokeWidth={2}
        />
        <text x={chanceX} y={16} textAnchor="middle" fill="var(--color-ink-faint)">
          0.50 chance
        </text>

        {/* median rule for the selected condition */}
        <line
          x1={xScale(summary.median)}
          y1={plotTop - 4}
          x2={xScale(summary.median)}
          y2={height - 32}
          stroke="var(--color-signal)"
          strokeWidth={1.5}
          strokeDasharray="3 3"
        />

        {rows.map((r, i) => {
          const y = plotTop + 20 + i * rowGap;
          const xs = EXP4_CONDITIONS.map((c) => xScale(r.byCondition[c]));
          const minX = Math.min(...xs);
          const maxX = Math.max(...xs);
          const selectedX = xScale(r.byCondition[selected]);
          return (
            <g key={r.pair_id} transform={`translate(0, ${y})`}>
              <text x={marginLeft - 10} y={4} textAnchor="end" fill="var(--color-ink-soft)">
                {r.pair_id.replace(/_/g, " / ")}
              </text>
              <line x1={minX} y1={0} x2={maxX} y2={0} stroke="var(--color-hairline)" strokeWidth={1} />
              {EXP4_CONDITIONS.filter((c) => c !== selected).map((c) => (
                <line
                  key={c}
                  x1={xScale(r.byCondition[c])}
                  y1={-4}
                  x2={xScale(r.byCondition[c])}
                  y2={4}
                  stroke="var(--color-hairline-strong)"
                  strokeWidth={1}
                />
              ))}
              <g
                style={{
                  transform: `translateX(${selectedX}px)`,
                  transition: reducedMotion ? "none" : "transform 180ms ease",
                }}
              >
                <circle cx={0} cy={0} r={5} fill="var(--color-ink)" />
              </g>
            </g>
          );
        })}

        <text x={width - marginRight} y={height - 8} textAnchor="end" fill="var(--color-ink-faint)">
          AUROC
        </text>
      </svg>

      <figcaption className="mt-4 font-mono text-[12px] text-ink-soft">
        {CONDITION_LABEL[selected]} selected: median {summary.median.toFixed(4)}, {summary.nAboveChance} of{" "}
        {summary.n} dyads above chance. The 0.50 chance line is the majority-class baseline, measured for every
        dyad -- not a drawing convention.
      </figcaption>
    </figure>
  );
}
