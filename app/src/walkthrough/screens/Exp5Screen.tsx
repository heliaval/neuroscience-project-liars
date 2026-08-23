import { LearningCurve } from "@/components/LearningCurve";
import { TechnicalDetails } from "@/components/TechnicalDetails";
import { getClaim, getExp5AggregateNote, getExp5Curve, getPairedTest } from "@/data/selectors";
import { Caveat, WalkthroughScreen } from "../WalkthroughScreen";

export function Exp5Screen({ revealStep }: { revealStep: number }) {
  const claim = getClaim("exp5_learning_curve");
  const test = getPairedTest("exp5_learning_curve");
  const points = getExp5Curve();
  const note = getExp5AggregateNote();

  return (
    <WalkthroughScreen
      eyebrow="Experiment 5 · Does history accumulate?"
      claim={claim.plain_language}
      revealStep={revealStep}
      evidence={<LearningCurve points={points} reveal={revealStep >= 1} />}
      caveat={
        <>
          <Caveat>
            The picture is not the test. The results file labels this curve{" "}
            <span className="font-mono text-[0.9em] text-ink">{note}</span>. The test is a paired comparison
            across {test.n} pairs: median Δ AUROC {test.median_delta.toFixed(4)}, {test.n_positive} of {test.n}{" "}
            positive, sign p {test.sign_test_p.toFixed(4)}, permutation p {test.permutation_p.toFixed(4)}, 95%
            CI [{test.ci95.lower?.toFixed(4)}, {test.ci95.upper?.toFixed(4)}]. Read the numbers, not the slope.
          </Caveat>
          <TechnicalDetails text={claim.technical} />
        </>
      }
    />
  );
}
