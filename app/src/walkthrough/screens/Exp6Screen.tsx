import { EarlyLateSlope } from "@/components/EarlyLateSlope";
import { TechnicalDetails } from "@/components/TechnicalDetails";
import {
  getClaim,
  getExp6EarlyLate,
  getExp6GateVerdict,
  getExp6ReportingConstraint,
  getPairedTest,
} from "@/data/selectors";
import { Caveat, WalkthroughScreen } from "../WalkthroughScreen";

export function Exp6Screen({ revealStep }: { revealStep: number }) {
  const claim = getClaim("exp6_observer_early_late");
  const test = getPairedTest("exp6_observer_early_late");
  const pairs = getExp6EarlyLate();
  const gate = getExp6GateVerdict();

  return (
    <WalkthroughScreen
      eyebrow="Experiment 6 · The observer's EEG, early vs late"
      claim={claim.plain_language}
      revealStep={revealStep}
      evidence={<EarlyLateSlope pairs={pairs} reveal={revealStep >= 1} />}
      caveat={
        <>
          <Caveat>
            <span className="block text-[0.9em] leading-[1.45]">
              Each line is one pair, from its early block to its late block. Six of the ten rise, four fall, and
              the median change is {test.median_delta.toFixed(4)}, sign p {test.sign_test_p.toFixed(4)},
              permutation p {test.permutation_p.toFixed(4)}. One structural point the chart makes easy to
              misread: the observing participant is not the same person in both blocks. Roles switch between
              the halves, so each line runs from one participant's early block to their partner's late block,
              not from one person's start to their finish. The power gate is also split across this
              experiment's sub-questions:{" "}
              {Object.entries(gate).map(([k, v], i) => (
                <span key={k}>
                  {i > 0 && ", "}
                  <span className="font-mono text-[0.9em] text-ink">
                    {k} {v}
                  </span>
                </span>
              ))}
              .
            </span>
          </Caveat>
          <div className="mt-3 border border-hairline-strong bg-paper-raised p-3">
            <p className="font-sans text-[11px] uppercase tracking-[0.08em] text-ink-faint">
              Binding reporting constraint (S16)
            </p>
            <p className="mt-1.5 font-mono text-[11px] leading-snug text-ink-soft">
              {getExp6ReportingConstraint()}
            </p>
          </div>
          <TechnicalDetails text={claim.technical} />
        </>
      }
    />
  );
}
