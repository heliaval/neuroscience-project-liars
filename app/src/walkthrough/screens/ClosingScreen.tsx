import { Link } from "react-router-dom";
import { motion, useReducedMotion } from "motion/react";
import { ProvenanceBanner } from "@/components/ProvenanceBanner";
import { WT, itemVariants, stageVariants } from "../motion";

export function ClosingScreen({ revealStep }: { revealStep: number }) {
  void revealStep; // no reveal beat
  const reduced = useReducedMotion();
  const itemTransition = reduced ? { duration: 0 } : undefined;

  return (
    <motion.div
      initial="hidden"
      animate="shown"
      custom={reduced ? 0 : WT.stagger}
      variants={stageVariants}
      className="mx-auto w-full max-w-[92ch] px-6 py-12 sm:px-8"
    >
      <motion.p
        variants={itemVariants}
        transition={itemTransition}
        className="font-sans text-[13px] uppercase tracking-[0.08em] text-ink-soft"
      >
        What the seven experiments came to
      </motion.p>

      <motion.h2
        variants={itemVariants}
        transition={itemTransition}
        className="mt-5 max-w-[46ch] font-serif text-[1.9rem] font-normal leading-[1.25] text-ink sm:text-[2.4rem]"
      >
        Nothing generalized to unseen pairs. The relationship-specific hypothesis, the one this project was
        built around, went the wrong way. The single comparison that reached significance was a negative one.
      </motion.h2>

      <motion.div variants={itemVariants} transition={itemTransition} className="mt-10 border border-hairline">
        <ProvenanceBanner />
      </motion.div>

      <motion.p
        variants={itemVariants}
        transition={itemTransition}
        className="mt-10 font-serif text-[1.05rem] leading-[1.6] text-ink-soft"
      >
        The per-dyad detail, the condition comparison in levels, and the sections still running on placeholder
        data are all on the{" "}
        <Link
          to="/dashboard"
          className="text-signal underline decoration-signal-soft decoration-2 underline-offset-4 hover:decoration-signal"
        >
          full dashboard
        </Link>
        .
      </motion.p>
    </motion.div>
  );
}
