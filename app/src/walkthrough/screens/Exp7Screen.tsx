import { RankedCIBars } from "@/components/RankedCIBars";
import { TechnicalDetails } from "@/components/TechnicalDetails";
import {
  getClaim,
  getExp7BhCorrectedCount,
  getExp7InterbrainVsDeceiver,
  getExp7Ranked,
} from "@/data/selectors";
import { Caveat, WalkthroughScreen } from "../WalkthroughScreen";

export function Exp7Screen({ revealStep }: { revealStep: number }) {
  const claim = getClaim("exp7_observer_vs_deceiver");
  const sets = getExp7Ranked();
  const interbrain = getExp7InterbrainVsDeceiver();
  const bhCount = getExp7BhCorrectedCount();

  return (
    <WalkthroughScreen
      eyebrow="Experiment 7 · One brain or two?"
      claim={claim.plain_language}
      revealStep={revealStep}
      evidence={<RankedCIBars sets={sets} reveal={revealStep >= 1} />}
      caveat={
        <>
          <Caveat>
            The one comparison here that reaches nominal significance points the wrong way. Inter-brain features
            alone, against the deceiver's own EEG: median Δ AUROC {interbrain.median_delta.toFixed(4)},{" "}
            {interbrain.n_positive} of {interbrain.n} pairs positive, permutation p{" "}
            {interbrain.permutation_p.toFixed(4)}, 95% CI [{interbrain.ci95.lower?.toFixed(4)},{" "}
            {interbrain.ci95.upper?.toFixed(4)}]. Dyadic coupling features do not carry the deception signal;
            they carry measurably less of it than one participant's EEG does. This comparison is designated
            confirmatory-secondary, so it sits outside the {bhCount}-comparison exploratory family in{" "}
            <span className="font-mono text-[0.9em] text-ink">tests_exploratory_bh</span>, which holds the
            Benjamini-Hochberg-corrected values for the exploratory contrasts.
          </Caveat>
          <TechnicalDetails text={claim.technical} />
        </>
      }
    />
  );
}
