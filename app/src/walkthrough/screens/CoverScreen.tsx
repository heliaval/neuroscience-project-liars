import { motion, useReducedMotion } from "motion/react";
import { WT, itemVariants, stageVariants } from "../motion";
import { getDyads, getFrozenHypotheses, getGeneratedAt, getParticipantCount } from "@/data/selectors";

const HYPOTHESIS_LABEL: Record<string, string> = {
  H1: "Universal",
  H2: "Person-specific",
  H3: "Relationship-specific",
};

export function CoverScreen({ revealStep }: { revealStep: number }) {
  void revealStep; // the cover has no reveal beat; it animates once on mount
  const reduced = useReducedMotion();
  const hypotheses = getFrozenHypotheses();
  const dyadCount = getDyads().length;
  const participantCount = getParticipantCount();
  const generatedAt = getGeneratedAt().slice(0, 10);
  const itemTransition = reduced ? { duration: 0 } : undefined;

  return (
    <motion.div
      initial="hidden"
      animate="shown"
      custom={reduced ? 0 : WT.stagger}
      variants={stageVariants}
      className="mx-auto w-full max-w-[92ch] px-8 py-6 sm:px-12 lg:px-16 xl:px-20"
    >
      <motion.p
        variants={itemVariants}
        transition={itemTransition}
        className="font-sans text-[13px] uppercase tracking-[0.08em] text-ink-soft"
      >
        Can your brain learn a liar
      </motion.p>

      <motion.h1
        variants={itemVariants}
        transition={itemTransition}
        className="mt-3 max-w-[26ch] font-serif text-[2rem] font-normal leading-[1.12] text-ink sm:text-[2.6rem]"
      >
        We tested whether deception has a universal EEG signature — or whether repeated interaction
        makes it specific to one opponent.
      </motion.h1>

      <motion.p
        variants={itemVariants}
        transition={itemTransition}
        className="mt-4 max-w-[64ch] font-serif text-[1rem] leading-[1.5] text-ink-soft"
      >
        <span className="font-mono text-[0.95rem] text-ink">{dyadCount}</span> pairs (
        <span className="font-mono text-[0.95rem] text-ink">{participantCount}</span> participants) played a
        repeated deception game with simultaneous EEG. One participant chose whether to lie about a decision;
        the other tried to call it. Roles switched across sessions, so every participant appears as both.
      </motion.p>

      <motion.p
        variants={itemVariants}
        transition={itemTransition}
        className="mt-5 font-sans text-[13px] uppercase tracking-[0.08em] text-ink-soft"
      >
        Three hypotheses, frozen before any model was fit
      </motion.p>

      <motion.ol variants={itemVariants} transition={itemTransition} className="mt-3 max-w-[76ch] space-y-2">
        {["H1", "H2", "H3"].map((id) => (
          <li key={id} className="border-l-2 border-hairline-strong pl-4">
            <span className="font-mono text-[12px] text-ink">
              {id} · {HYPOTHESIS_LABEL[id]}
            </span>
            <p className="mt-0.5 font-serif text-[0.95rem] leading-[1.4] text-ink-soft">{hypotheses[id]}</p>
          </li>
        ))}
      </motion.ol>

      <motion.p
        variants={itemVariants}
        transition={itemTransition}
        className="mt-5 font-mono text-[11px] text-ink-soft"
      >
        Every number in the seven screens that follow is read from results/results.v1.json, generated{" "}
        {generatedAt} by this project's own pipeline. Nothing on these screens is illustrative.
      </motion.p>
    </motion.div>
  );
}
