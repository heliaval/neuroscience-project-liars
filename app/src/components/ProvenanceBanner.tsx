import { getPlaceholderSections, getRealSections, getMixedSections, getProvenance, type SectionId } from "@/data/selectors";

const SECTION_LABELS: Record<SectionId, string> = {
  gate: "power gate",
  frozen: "frozen hypotheses",
  dyads: "dyad roster",
  exp1: "Experiment 1 (pooled baseline)",
  exp2: "Experiment 2 (leave-one-dyad-out)",
  exp3: "Experiment 3 (person-specific)",
  exp4: "Experiment 4 (dyad-specific)",
  exp5: "Experiment 5 (learning curve)",
  exp6: "Experiment 6 (observer early/late)",
  exp7: "Experiment 7 (one brain vs two)",
  tests: "statistical claims",
};

/** Page-level notice. Renders whenever any section is still placeholder or mixed --
 * not gated on meta.is_fixture, since the real-results swap left is_fixture false
 * while the dyad fingerprint columns are still invented. interpretability,
 * interbrain, trials, and failures (the four sections that used to be invented
 * here) were dropped from the contract entirely rather than fixed, since nothing
 * in the app ever rendered them -- see PROGRESS.md. Disappears once the only
 * remaining placeholder (dyad fingerprints) reaches "real" too. */
export function ProvenanceBanner() {
  const placeholder = getPlaceholderSections();
  const mixed = getMixedSections();
  const real = getRealSections();

  if (placeholder.length === 0 && mixed.length === 0) return null;

  return (
    <div className="border-b border-hairline-strong bg-paper-raised">
      <div className="mx-auto max-w-[68ch] px-6 py-3 font-sans text-[13px] leading-snug text-ink-soft sm:px-8">
        <p>
          <span className="font-medium text-ink">Parts of this page are still placeholder data.</span>{" "}
          Real, computed results:{" "}
          {real.map((s, i) => (
            <span key={s}>
              {i > 0 && ", "}
              <span className="font-mono text-[12px]">{SECTION_LABELS[s]}</span>
            </span>
          ))}
          .{" "}
          {mixed.length > 0 && (
            <>
              Partly real, partly placeholder:{" "}
              {mixed.map((s, i) => (
                <span key={s}>
                  {i > 0 && ", "}
                  <span className="font-mono text-[12px]">{SECTION_LABELS[s]}</span>
                </span>
              ))}
              .{" "}
            </>
          )}
          {placeholder.length > 0 && (
            <>
              Invented placeholder numbers, constrained to a plausible range but not measured:{" "}
              {placeholder.map((s, i) => (
                <span key={s}>
                  {i > 0 && ", "}
                  <span className="font-mono text-[12px]">{SECTION_LABELS[s]}</span>
                </span>
              ))}
              .
            </>
          )}
        </p>
      </div>
    </div>
  );
}

/** Per-figure marker any component can drop in next to a specific number or chart. */
export function ProvenanceMark({ section }: { section: SectionId }) {
  const provenance = getProvenance(section);
  if (provenance === "real") return null;

  const label = provenance === "mixed" ? "partly placeholder" : "placeholder";
  return (
    <span className="inline-flex items-center gap-1 rounded-[var(--radius-editorial)] border border-hairline-strong px-1.5 py-0.5 font-sans text-[11px] uppercase tracking-wide text-ink-faint">
      {label}
    </span>
  );
}
