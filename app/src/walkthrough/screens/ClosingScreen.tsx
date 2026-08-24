import { Link } from "react-router-dom";
import { motion, useReducedMotion } from "motion/react";
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
      className="flex h-full min-h-0 w-full flex-1 flex-col items-center justify-center bg-[#10141a] px-8 py-10 text-center sm:px-12"
    >
      <motion.p
        variants={itemVariants}
        transition={itemTransition}
        className="font-sans text-[13px] uppercase tracking-[0.14em] text-[#7fa0bd]"
      >
        Seven experiments later
      </motion.p>

      <motion.h2
        variants={itemVariants}
        transition={itemTransition}
        className="mt-6 max-w-[24ch] font-serif text-[2.4rem] font-normal leading-[1.15] text-white sm:text-[3.4rem]"
      >
        The relationship-specific signature we built this project to find never showed up.
      </motion.h2>

      <motion.p
        variants={itemVariants}
        transition={itemTransition}
        className="mt-8 max-w-[52ch] font-serif text-[1.15rem] leading-[1.6] text-white/75"
      >
        Nothing generalized to a pair the model hadn't already seen, and the one comparison that did reach
        significance pointed the wrong way.
      </motion.p>

      <motion.p
        variants={itemVariants}
        transition={itemTransition}
        className="mt-14 font-serif text-[1.6rem] font-normal text-[#8fb4d6] sm:text-[2rem]"
      >
        Thank you for following all seven through to the end.
      </motion.p>

      <motion.p
        variants={itemVariants}
        transition={itemTransition}
        className="mt-10 font-sans text-[12px] text-white/45"
      >
        Per-dyad detail, condition comparisons, and technical notes are on the{" "}
        <Link to="/dashboard" className="text-[#7fa0bd] underline decoration-[#7fa0bd]/50 underline-offset-4 hover:decoration-[#7fa0bd]">
          full dashboard
        </Link>
        .
      </motion.p>
    </motion.div>
  );
}
