import type { ReactNode } from "react";
import { motion, useReducedMotion } from "motion/react";
import { WT, itemVariants, stageVariants } from "./motion";

export function WalkthroughScreen({
  eyebrow,
  claim,
  revealStep,
  evidence,
  caveat,
  secondBeat,
  emphasis = false,
}: {
  eyebrow: string;
  claim: string;
  revealStep: number;
  evidence: ReactNode;
  caveat: ReactNode;
  secondBeat?: ReactNode;
  emphasis?: boolean;
}) {
  const reduced = useReducedMotion();
  const revealed = revealStep >= 1;
  const stagger = reduced ? 0 : emphasis ? WT.staggerSlow : WT.stagger;

  // itemVariants' own transition is a fixed duration -- override it here so a
  // reduced-motion reader gets an instant reveal on the evidence/caveat blocks too,
  // not just a shorter stagger between them.
  const itemTransition = reduced ? { duration: 0 } : undefined;

  return (
    <div className="w-full px-6 py-4 sm:px-10 lg:px-16 xl:px-20">
      <p className="font-sans text-[13px] uppercase tracking-[0.08em] text-ink-soft">{eyebrow}</p>

      {/* Claim and evidence sit side by side on wide viewports, using the width a
          single centered text column left empty, instead of the evidence piling up
          underneath the claim. The claim keeps a fixed reading-width column; the
          evidence fills the space beside it as it mounts on reveal. */}
      <div className="mt-3 grid grid-cols-1 items-start gap-x-16 gap-y-4 lg:grid-cols-[minmax(0,46ch)_minmax(0,1fr)]">
        <h2
          className={`font-serif font-normal leading-[1.2] text-ink ${
            emphasis ? "text-[1.9rem] sm:text-[2.4rem]" : "text-[1.7rem] sm:text-[2.1rem]"
          }`}
        >
          {claim}
        </h2>

        <motion.div initial="hidden" animate={revealed ? "shown" : "hidden"} custom={stagger} variants={stageVariants}>
          {revealed && (
            <motion.div variants={itemVariants} transition={itemTransition}>
              {evidence}
            </motion.div>
          )}
        </motion.div>
      </div>

      <motion.div
        initial="hidden"
        animate={revealed ? "shown" : "hidden"}
        custom={stagger}
        variants={stageVariants}
        className="mt-6 max-w-[70ch]"
      >
        {revealed && (
          <motion.div variants={itemVariants} transition={itemTransition}>
            {caveat}
          </motion.div>
        )}
      </motion.div>

      {secondBeat && revealStep >= 2 && (
        <motion.div
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: reduced ? 0 : WT.slow, ease: WT.ease }}
          className="mt-4 border-t border-hairline pt-4"
        >
          {secondBeat}
        </motion.div>
      )}
    </div>
  );
}

/** Caveats arrive with the evidence, never before it. Same hairline-rule idiom as
 * TechnicalDetails, one level louder because these are load-bearing. */
export function Caveat({ children }: { children: ReactNode }) {
  return (
    <div className="border-l-2 border-hairline-strong pl-4 font-serif text-[1rem] leading-[1.6] text-ink-soft">
      {children}
    </div>
  );
}
