import { useState } from "react";
import { ConditionComparison } from "@/components/ConditionComparison";
import { TechnicalDetails } from "@/components/TechnicalDetails";
import { ProvenanceMark } from "@/components/ProvenanceBanner";
import { EXP4_CONDITIONS, getConditionSummary, getExp4HeldOut, getExperiment, type Exp4ConditionId } from "@/data/selectors";

const CONDITION_LABEL: Record<Exp4ConditionId, string> = {
  universal: "Stranger",
  person_specific: "Person-specific",
  dyad_specific: "Dyad-specific",
};

const CONDITION_DESCRIPTION: Record<Exp4ConditionId, string> = {
  universal: "Trained on every other dyad's trials. Has never seen this person or this pair.",
  person_specific: "Trained only on the tested participant's own earlier trials.",
  dyad_specific: "Those same trials, plus the partner's entire first-session block.",
};

function formatTrainRange(min: number, max: number): string {
  const fmt = (n: number) => n.toLocaleString("en-US");
  return min === max ? fmt(min) : `${fmt(min)}–${fmt(max)}`;
}

function ConditionColumn({
  cond,
  isSelected,
  onSelect,
}: {
  cond: Exp4ConditionId;
  isSelected: boolean;
  onSelect: () => void;
}) {
  const summary = getConditionSummary(cond);
  return (
    <button
      type="button"
      role="radio"
      aria-checked={isSelected}
      onClick={onSelect}
      className={`flex-1 border-t-2 px-3 py-3 text-left transition-colors focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-signal ${
        isSelected ? "border-signal" : "border-hairline hover:border-hairline-strong"
      }`}
    >
      <div
        className={`font-sans text-[12px] font-medium uppercase tracking-wide ${
          isSelected ? "text-ink" : "text-ink-faint"
        }`}
      >
        {CONDITION_LABEL[cond]}
      </div>
      <p className="mt-1.5 font-serif text-[0.9rem] leading-snug text-ink-soft">{CONDITION_DESCRIPTION[cond]}</p>
      <div className="mt-2 font-mono text-[11px] text-ink-faint">
        n_train {formatTrainRange(summary.nTrainMin, summary.nTrainMax)}
      </div>
      <div className={`mt-1 font-mono text-[1.1rem] ${isSelected ? "text-ink" : "text-ink-soft"}`}>
        {summary.median.toFixed(4)}
      </div>
    </button>
  );
}

export function StrangerVsFamiliar() {
  const [selected, setSelected] = useState<Exp4ConditionId>("universal");
  const held = getExp4HeldOut();
  const exp4 = getExperiment("exp4");
  const volumeControlled = exp4.tests.dyad_gain_volume_controlled;
  const matched = exp4.tests.dyad_gain_matched;

  const index = EXP4_CONDITIONS.indexOf(selected);
  const move = (delta: number) => {
    const next = (index + delta + EXP4_CONDITIONS.length) % EXP4_CONDITIONS.length;
    setSelected(EXP4_CONDITIONS[next]);
  };

  return (
    <section className="mx-auto max-w-[68ch] px-6 py-16 sm:px-8">
      <div className="mb-2 flex items-center gap-2">
        <h2 className="font-serif text-[1.5rem] font-normal text-ink">Does knowing them help?</h2>
        <ProvenanceMark section="exp4" />
      </div>

      <p className="mt-3 font-serif text-[1.05rem] leading-[1.6] text-ink-soft">
        Three models, fit identically, tested on the same held-out rows: each pair's last {held.nTest} trials, seq{" "}
        {held.seqMin}–{held.seqMax}, which none of the three ever saw during training. The only thing that changes
        between them is where the training data came from.
      </p>

      <div
        role="radiogroup"
        aria-label="Training condition"
        className="mt-8 flex divide-x divide-hairline border-b border-hairline"
        onKeyDown={(e) => {
          if (e.key === "ArrowRight" || e.key === "ArrowDown") {
            e.preventDefault();
            move(1);
          } else if (e.key === "ArrowLeft" || e.key === "ArrowUp") {
            e.preventDefault();
            move(-1);
          }
        }}
      >
        {EXP4_CONDITIONS.map((cond) => (
          <ConditionColumn key={cond} cond={cond} isSelected={cond === selected} onSelect={() => setSelected(cond)} />
        ))}
      </div>

      <ConditionComparison selected={selected} />

      <p className="mt-3 font-serif text-[1.05rem] leading-[1.6] text-ink-soft">
        All three sit within a few points of chance. The dyad-specific model — the one this project's central
        hypothesis predicted would win — is the lowest of the three.
      </p>

      {volumeControlled && matched && (
        <TechnicalDetails
          text={`The obvious objection: dyad-specific just gets more training rows. Two controls hold that constant. Volume-matched (same n_train as dyad-specific, drawn from other dyads): median ΔAUROC ${volumeControlled.median_delta.toFixed(4)}. Source-matched (same participant's rows, drawn from a different dyad): median ΔAUROC ${matched.median_delta.toFixed(4)}. Both negative -- the result survives both controls.`}
        />
      )}

      <p className="mt-6 font-serif text-[1.05rem] leading-[1.6] text-ink-soft">
        The levels are close. The next section shows the difference itself, pair by pair, with the test that
        consumes it.
      </p>
    </section>
  );
}
