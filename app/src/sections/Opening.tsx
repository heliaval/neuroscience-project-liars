import { getDyads, getParticipantCount } from "@/data/selectors";

export function Opening() {
  const dyadCount = getDyads().length;
  const participantCount = getParticipantCount();

  return (
    <section className="mx-auto max-w-[68ch] px-6 pt-16 pb-20 sm:px-8 sm:pt-24">
      <p className="font-sans text-[13px] uppercase tracking-[0.08em] text-ink-faint">
        Can your brain learn a liar
      </p>

      <h1 className="mt-4 font-serif text-[2.1rem] font-normal leading-[1.15] text-ink sm:text-[2.6rem]">
        We tested whether deception has a universal EEG signature — or
        whether repeated interaction causes neural responses to become
        specific to an opponent.
      </h1>

      <p className="mt-8 font-serif text-[1.1rem] leading-[1.6] text-ink-soft">
        <span className="font-mono text-[0.95rem] text-ink">{dyadCount}</span>{" "}
        pairs ({" "}
        <span className="font-mono text-[0.95rem] text-ink">{participantCount}</span>{" "}
        participants) played a repeated deception game while their EEG was
        recorded simultaneously. Each session paired a deceiver, choosing
        whether to lie about a decision, with an observer trying to detect it.
        Roles switched across sessions, so every participant appears in the
        data as both.
      </p>

      <p className="mt-6 font-serif text-[1.1rem] leading-[1.6] text-ink-soft">
        Three explanations compete: deception looks the same in every brain
        (universal), it looks different from person to person (person
        specific), or it looks different from one relationship to the next,
        shaped by who you are lying to (dyad specific). The rest of this page
        walks through what the data actually shows, including where the
        central hypothesis did not hold up.
      </p>
    </section>
  );
}
