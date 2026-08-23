/** Both progress affordances, together: the numeric counter over the seven evidence
 * screens, and one hairline tick per screen (nine, bookends drawn shorter). */
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
    <footer className="sticky bottom-0 z-10 border-t border-hairline bg-paper">
      <div className="mx-auto flex w-full max-w-[92ch] items-center gap-6 px-6 py-4 sm:px-8">
        <button
          type="button"
          onClick={onBack}
          disabled={!canBack}
          className="font-sans text-[13px] text-signal underline decoration-signal-soft decoration-2 underline-offset-4 hover:decoration-signal disabled:cursor-default disabled:text-ink-faint disabled:no-underline"
        >
          Back
        </button>

        <div className="flex flex-1 items-end gap-1.5" aria-hidden="true">
          {Array.from({ length: screenCount }, (_, i) => {
            const isBookend = i === 0 || i === screenCount - 1;
            const passed = i <= screenIndex;
            return (
              <span
                key={i}
                className="flex-1"
                style={{
                  height: isBookend ? 4 : 10,
                  // ink-faint (not hairline) so the un-reached ticks clear 3:1 contrast
                  // against paper -- a progress indicator is a graphical object, not decoration.
                  background: passed ? "var(--color-ink)" : "var(--color-ink-faint)",
                  transition: `background var(--wt-dur-fast) var(--wt-ease)`,
                }}
              />
            );
          })}
        </div>

        <span
          className="min-w-[4ch] text-right font-mono text-[12px] text-ink-soft"
          aria-label={evidenceIndex === null ? undefined : `Evidence screen ${evidenceIndex} of ${evidenceCount}`}
        >
          {evidenceIndex === null ? "" : `${evidenceIndex} / ${evidenceCount}`}
        </span>

        <button
          type="button"
          onClick={onAdvance}
          disabled={!canAdvance}
          className="border border-hairline-strong px-4 py-1.5 font-sans text-[13px] text-ink hover:border-ink disabled:cursor-default disabled:border-hairline disabled:text-ink-faint"
        >
          {advanceLabel}
        </button>
      </div>
    </footer>
  );
}
