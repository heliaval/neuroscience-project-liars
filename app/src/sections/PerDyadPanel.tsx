import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import { PairedDotPlot } from "@/components/PairedDotPlot";
import { TechnicalDetails } from "@/components/TechnicalDetails";
import { ProvenanceMark } from "@/components/ProvenanceBanner";
import { getClaim, getPairedTest, getDyads, getAmendments, getExperiment, type ClaimId } from "@/data/selectors";

const CLAIM_TABS: { id: ClaimId; label: string }[] = [
  { id: "h3_dyad_gain", label: "Dyad-specific gain" },
  { id: "h2_person_gain", label: "Person-specific gain" },
  { id: "nfi_distribution", label: "Net familiarity index" },
];

function ClaimPanel({ id }: { id: ClaimId }) {
  const claim = getClaim(id);
  const test = getPairedTest(id);
  return (
    <TabsContent value={id}>
      <p className="font-serif text-[1.05rem] leading-[1.6] text-ink">{claim.plain_language}</p>
      <PairedDotPlot test={test} />
      <TechnicalDetails text={claim.technical} />
    </TabsContent>
  );
}

function DyadFingerprints() {
  const dyads = getDyads();
  return (
    <div className="mt-4">
      <div className="mb-3 flex items-center justify-between">
        <h3 className="font-sans text-[13px] font-medium uppercase tracking-wide text-ink-faint">
          Per-dyad fingerprint
        </h3>
        <ProvenanceMark section="dyads" />
      </div>
      <div className="overflow-x-auto">
        <table className="w-full min-w-[560px] border-collapse font-mono text-[12px]">
          <thead>
            <tr className="border-b border-hairline-strong text-left text-ink-faint">
              <th className="py-2 pr-4 font-sans font-medium normal-case">Pair</th>
              <th className="py-2 pr-4 text-right">Frontal coupling</th>
              <th className="py-2 pr-4 text-right">Temporal lag (ms)</th>
              <th className="py-2 pr-4 text-right">Alpha synchrony</th>
              <th className="py-2 pr-4 text-right">Observer response</th>
              <th className="py-2 text-right">Personalization gain</th>
            </tr>
          </thead>
          <tbody>
            {dyads.map((d) => (
              <tr key={d.pair_id} className="border-b border-hairline text-ink-soft">
                <td className="py-1.5 pr-4 font-sans text-ink">{d.pair_id.replace(/_/g, " / ")}</td>
                <td className="py-1.5 pr-4 text-right">{d.fingerprint.frontal_coupling.toFixed(3)}</td>
                <td className="py-1.5 pr-4 text-right">{d.fingerprint.temporal_lag_ms}</td>
                <td className="py-1.5 pr-4 text-right">{d.fingerprint.alpha_synchrony.toFixed(3)}</td>
                <td className="py-1.5 pr-4 text-right">{d.fingerprint.observer_response.toFixed(3)}</td>
                <td className="py-1.5 text-right">{d.fingerprint.personalization_gain.toFixed(4)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

export function PerDyadPanel() {
  const amendments = getAmendments();
  const exp4 = getExperiment("exp4");
  const amendment = amendments.find((a) => a.applies_to === "exp4");

  return (
    <section className="mx-auto max-w-[68ch] px-6 py-16 sm:px-8">
      <div className="mb-2 flex items-center gap-2">
        <h2 className="font-serif text-[1.5rem] font-normal text-ink">The per-dyad result</h2>
        <ProvenanceMark section="exp4" />
      </div>

      <p className="mt-3 font-serif text-[1.05rem] leading-[1.6] text-ink-soft">
        The levels above are close. Here is the difference itself — the same three comparisons, now paired
        dyad by dyad against a zero line, with the test that decides whether any of it is real. Each dot below
        is one pair.
      </p>

      <p className="mt-3 font-serif text-[1.05rem] leading-[1.6] text-ink-soft">
        This is <span className="font-mono text-[0.95em]">{exp4.per_dyad.length}</span> dots, not{" "}
        <span className="font-mono text-[0.95em]">{getDyads().length}</span>.
        {amendment && (
          <>
            {" "}
            <span className="text-ink">{amendment.excluded_unit?.replace(/_/g, " / ")}</span>{" "}
            is excluded from this comparison: {amendment.rationale.split(".")[0].toLowerCase()}.
          </>
        )}
      </p>

      <Tabs defaultValue="h3_dyad_gain" className="mt-8">
        <TabsList>
          {CLAIM_TABS.map((t) => (
            <TabsTrigger key={t.id} value={t.id}>
              {t.label}
            </TabsTrigger>
          ))}
        </TabsList>
        {CLAIM_TABS.map((t) => (
          <ClaimPanel key={t.id} id={t.id} />
        ))}
      </Tabs>

      <DyadFingerprints />
    </section>
  );
}
