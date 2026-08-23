/** exp4 is the only screen whose advance label already reads as a full sentence
 * ("See the same result in levels"); the others are short enough to fold into
 * the "click to ..." hint pill directly. */
function pillLabel(advanceLabel: string): string {
  switch (advanceLabel) {
    case "Show the evidence":
      return "click to reveal";
    case "Next":
      return "click to continue";
    case "End":
      return "click to finish";
    default:
      return `click to ${advanceLabel.toLowerCase()}`;
  }
}

/** Both progress affordances, together, kept deliberately small: a short row of
 * ticks plus Back at the bottom-left, a single "click to ..." hint pill at the
 * bottom-right (same idea as a "press key to ..." corner hint, just naming the
 * click interaction this deck actually uses instead of a keypress). No bar
 * chrome -- no border, no background -- so it reads as a light corner marking,
 * not a UI bar competing with the evidence above it. */
export function WalkthroughFooter({
  screenIndex,
  screenCount,
  evidenceIndex,
  evidenceCount,
  canBack,
  canAdvance,
  onBack,
  onAdvance,
  advanceLabel,
}: {
  screenIndex: number;
  screenCount: number;
  evidenceIndex: number | null;
  evidenceCount: number;
  canBack: boolean;
  canAdvance: boolean;
  onBack: () => void;
  onAdvance: () => void;
  advanceLabel: string;
}) {
  return (
    <footer className="flex w-full items-center justify-between gap-4 px-4 py-2 sm:px-6">
      <div className="flex items-center gap-3">
        <button
          type="button"
          onClick={onBack}
          disabled={!canBack}
          className="font-sans text-[12px] text-signal underline decoration-signal-soft decoration-2 underline-offset-4 hover:decoration-signal disabled:cursor-default disabled:text-ink-faint disabled:no-underline"
        >
          Back
        </button>

        <div className="flex items-center gap-1" aria-hidden="true">
          {Array.from({ length: screenCount }, (_, i) => (
            <span
              key={i}
              className="h-[3px] w-3"
              style={{
                // ink-faint (not hairline) so the un-reached ticks clear 3:1 contrast
                // against paper -- a progress indicator is a graphical object, not decoration.
                background: i <= screenIndex ? "var(--color-ink)" : "var(--color-ink-faint)",
                transition: `background var(--wt-dur-fast) var(--wt-ease)`,
              }}
            />
          ))}
        </div>

        <span
          className="font-mono text-[11px] text-ink-faint"
          aria-label={evidenceIndex === null ? undefined : `Evidence screen ${evidenceIndex} of ${evidenceCount}`}
        >
          {evidenceIndex === null ? "" : `${evidenceIndex}/${evidenceCount}`}
        </span>
      </div>

      <button
        type="button"
        onClick={onAdvance}
        disabled={!canAdvance}
        className="rounded-[var(--radius-editorial)] border border-hairline px-2.5 py-1 font-mono text-[11px] lowercase tracking-wide text-ink-faint hover:border-hairline-strong hover:text-ink-soft disabled:cursor-default disabled:opacity-0"
      >
        {pillLabel(advanceLabel)}
      </button>
    </footer>
  );
}
