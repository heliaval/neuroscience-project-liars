import { isFixture, getPlaceholderSections, getRealSections, getProvenance, type SectionId } from "@/data/selectors";

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
  exp8: "Experiment 8 (information onset)",
  tests: "statistical claims",
  interpretability: "electrode/coefficient maps",
  interbrain: "inter-brain network",
  trials: "individual trial browser",
  failures: "failure case gallery",
};

/** Page-level notice. Disappears automatically the moment meta.is_fixture flips to
 * false on the real-results swap -- no code change required (plans/web-app-scaffold.md
 * §5.1). */
export function ProvenanceBanner() {
  if (!isFixture()) return null;

  const placeholder = getPlaceholderSections();
  const real = getRealSections();

  return (
    <div className="border-b border-hairline-strong bg-paper-raised">
      <div className="mx-auto max-w-[68ch] px-6 py-3 font-sans text-[13px] leading-snug text-ink-soft sm:px-8">
        <p>
          <span className="font-medium text-ink">This page is running on placeholder data.</span>{" "}
          Real, computed results:{" "}
          {real.map((s, i) => (
            <span key={s}>
              {i > 0 && ", "}
              <span className="font-mono text-[12px]">{SECTION_LABELS[s]}</span>
            </span>
          ))}
          . Invented placeholder numbers, constrained to a plausible range but not measured:{" "}
          {placeholder.map((s, i) => (
            <span key={s}>
              {i > 0 && ", "}
              <span className="font-mono text-[12px]">{SECTION_LABELS[s]}</span>
            </span>
          ))}
          .
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
