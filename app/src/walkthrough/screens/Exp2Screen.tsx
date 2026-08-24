import { PerDyadAurocPlot } from "@/components/PerDyadAurocPlot";
import { TypographicNumberReveal } from "@/components/TypographicNumberReveal";
import { TechnicalDetails } from "@/components/TechnicalDetails";
import {
  getClaim,
  getExp2DyadsAboveChance,
  getExp2PerDyadAuroc,
  getExp2VsExp1Delta,
  getHeadline,
  getPermutationNullSummary,
} from "@/data/selectors";
import { Caveat, WalkthroughScreen } from "../WalkthroughScreen";

export function Exp2Screen({ revealStep }: { revealStep: number }) {
  const claim = getClaim("h1_lodo_generalization");
  const headline = getHeadline("exp2");
  const nullSummary = getPermutationNullSummary("exp2");
  const scores = getExp2PerDyadAuroc();
  const dyads = getExp2DyadsAboveChance();
  const delta = getExp2VsExp1Delta();
  const revealed = revealStep >= 1;

  return (
    <WalkthroughScreen
      eyebrow="Experiment 2 · H1, universal signature"
      claim={claim.plain_language}
      revealStep={revealStep}
      evidence={
        <div className="flex flex-col gap-8 lg:min-h-0 lg:flex-1">
          <TypographicNumberReveal
            value={headline.value}
            lower={headline.lower}
            upper={headline.upper}
            reveal={revealed}
          />
          <PerDyadAurocPlot
            scores={scores}
            reference={headline.value}
            referenceLabel="pooled leave-one-dyad-out"
            reveal={revealed}
          />
        </div>
      }
      caveat={
        <>
          <Caveat>
            Each dot is one pair the model had never seen. {dyads.above} of {dyads.total} land above chance,
            which sounds like a signal until you notice how little separates them: the whole spread is roughly
            six points of AUROC around 0.50. Against the pooled Experiment 1 model, this loses{" "}
            {delta.toFixed(4)}. The permutation p is {nullSummary.pValue.toFixed(4)}, just past 0.05 rather than
            under it. This was a pre-registered confirmatory test, so the result stands as it fell: it was not
            re-run with a different model family, given a dropped fold, or reframed to favor one side.
          </Caveat>
          <TechnicalDetails text={claim.technical} />
        </>
      }
    />
  );
}
