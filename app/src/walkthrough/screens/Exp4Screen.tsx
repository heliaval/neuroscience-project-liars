import { ConditionComparison } from "@/components/ConditionComparison";
import { PairedDotPlot } from "@/components/PairedDotPlot";
import { TechnicalDetails } from "@/components/TechnicalDetails";
import { getClaim, getExp4Verdict, getPairedTest } from "@/data/selectors";
import { Caveat, WalkthroughScreen } from "../WalkthroughScreen";

export function Exp4Screen({ revealStep }: { revealStep: number }) {
  const claim = getClaim("h3_dyad_gain");
  const test = getPairedTest("h3_dyad_gain");
  const nfi = getPairedTest("nfi_distribution");
  const verdict = getExp4Verdict();

  return (
    <WalkthroughScreen
      emphasis
      eyebrow="Experiment 4 · H3, the project's main hypothesis"
      claim={claim.plain_language}
      revealStep={revealStep}
      evidence={
        // [&>figure] overrides PairedDotPlot's own my-8 (needed unmodified on /dashboard)
        // -- this screen is the one place two evidence charts share a viewport at once.
        <div className="[&>figure]:my-1!">
          <PairedDotPlot test={test} reveal={revealStep >= 1} maxHeight="min(24vh,230px)" />
        </div>
      }
      caveat={
        <>
          <Caveat>
            <span className="block text-[0.9em] leading-[1.45]">
              <span className="font-mono text-[0.95em] text-ink">{verdict}</span>. Two of this screen's numbers
              point different directions and both are reported as they fell: the 95% CI is [
              {test.ci95.lower?.toFixed(4)}, {test.ci95.upper?.toFixed(4)}], which excludes zero on the
              negative side, while the sign test over the same {test.n} pairs gives p{" "}
              {test.sign_test_p.toFixed(4)}, which does not reach significance. The permutation p is{" "}
              {test.permutation_p.toFixed(4)}. A combined relationship-plus-population model does not fix it
              either: the net familiarity index has a median of {nfi.median_delta.toFixed(4)} with{" "}
              {nfi.n_positive} of {nfi.n} pairs positive.
            </span>
          </Caveat>
          <TechnicalDetails text={claim.technical} />
        </>
      }
      secondBeat={
        <>
          <p className="font-sans text-[13px] uppercase tracking-[0.08em] text-ink-faint">
            The same result in levels
          </p>
          <p className="mt-2 max-w-[60ch] font-serif text-[0.95rem] leading-[1.4] text-ink-soft">
            Three models fit identically and tested on the same held-out rows. The dyad-specific model — the
            one the hypothesis predicted would win — is the lowest of the three.
          </p>
          <div className="[&>figure]:my-1!">
            <ConditionComparison selected="dyad_specific" maxHeight="min(24vh,230px)" />
          </div>
        </>
      }
    />
  );
}
