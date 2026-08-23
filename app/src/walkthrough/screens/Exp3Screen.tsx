import { PairedDotPlot } from "@/components/PairedDotPlot";
import { TechnicalDetails } from "@/components/TechnicalDetails";
import { getClaim, getExp3Exclusions, getPairedTest } from "@/data/selectors";
import { Caveat, WalkthroughScreen } from "../WalkthroughScreen";

export function Exp3Screen({ revealStep }: { revealStep: number }) {
  const claim = getClaim("exp3_personalized_vs_population");
  const test = getPairedTest("exp3_personalized_vs_population");
  const exclusions = getExp3Exclusions();

  return (
    <WalkthroughScreen
      eyebrow="Experiment 3 · H2, person-specific signature"
      claim={claim.plain_language}
      revealStep={revealStep}
      evidence={<PairedDotPlot test={test} reveal={revealStep >= 1} />}
      caveat={
        <>
          <Caveat>
            The 95% CI is [{test.ci95.lower?.toFixed(4)}, {test.ci95.upper?.toFixed(4)}] — it crosses zero, so
            a true effect of zero is not ruled out by this data. The comparison also runs on {test.n} pairs, not
            all twelve: {exclusions.sub01 && <span>{exclusions.sub01}</span>}{" "}
            {exclusions.sub02 && <span>{exclusions.sub02}</span>} That is a gap in the archive, not a property
            of those participants.
          </Caveat>
          <TechnicalDetails text={claim.technical} />
        </>
      }
    />
  );
}
