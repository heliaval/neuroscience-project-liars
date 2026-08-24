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
      className="mx-auto flex h-full min-h-0 w-full max-w-[1440px] flex-1 flex-col px-8 py-10 sm:px-12 lg:px-16 xl:px-20"
    >
      <motion.p
        variants={itemVariants}
        transition={itemTransition}
        className="font-sans text-[1.8rem] font-medium uppercase tracking-tight text-ink sm:text-[2.4rem]"
      >
        Conclusion
      </motion.p>

      <motion.h2
        variants={itemVariants}
        transition={itemTransition}
        className="mt-4 max-w-[36ch] font-serif text-[2.6rem] font-normal leading-[1.15] text-ink sm:text-[4.2rem]"
      >
        According to the EEG data, we achieved a null result.
      </motion.h2>

      <motion.p
        variants={itemVariants}
        transition={itemTransition}
        className="mt-8 max-w-[70ch] font-serif text-[1.15rem] leading-[1.6] text-ink-soft"
      >
        We started with a simple question: does lying leave a mark on the brain that a model can pick up. A
        model trained across everyone did just barely better than a coin flip. It didn't hold up on a pair it
        had never seen before. Personalizing the model to one person didn't help, and neither did learning a
        specific relationship, which was the whole idea behind this project. More training data didn't change
        the picture, the observer's brain didn't get any sharper over the course of a session, and no
        combination of inputs told a different story.
      </motion.p>

      <motion.p
        variants={itemVariants}
        transition={itemTransition}
        className="mt-auto self-end font-serif text-[1.4rem] font-normal text-ink sm:text-[1.7rem]"
      >
        Thanks.
      </motion.p>
    </motion.div>
  );
}
