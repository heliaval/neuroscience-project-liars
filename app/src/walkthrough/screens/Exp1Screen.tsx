import { NullHistogram } from "@/components/NullHistogram";
import { TypographicNumberReveal } from "@/components/TypographicNumberReveal";
import { TechnicalDetails } from "@/components/TechnicalDetails";
import { getClaim, getExp1Caveat, getHeadline, getPermutationNullSummary } from "@/data/selectors";
import { Caveat, WalkthroughScreen } from "../WalkthroughScreen";

export function Exp1Screen({ revealStep }: { revealStep: number }) {
  const claim = getClaim("exp1_above_chance");
  const headline = getHeadline("exp1");
  const nullSummary = getPermutationNullSummary("exp1");
  const revealed = revealStep >= 1;

  return (
    <WalkthroughScreen
      eyebrow="Experiment 1 · Pooled baseline"
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
          <NullHistogram
            distribution={nullSummary.distribution}
            observed={nullSummary.observed}
            band={nullSummary.band}
            pValue={nullSummary.pValue}
            nPermutations={nullSummary.nPermutations}
            reveal={revealed}
          />
        </div>
      }
      caveat={
        <>
          <Caveat>{getExp1Caveat()}</Caveat>
          <TechnicalDetails text={claim.technical} />
        </>
      }
    />
  );
}
