import { motion, useReducedMotion } from "motion/react";
import { WT } from "@/walkthrough/motion";

/**
 * Screens 1 and 2 each carry one headline number, so their number reveal is typographic:
 * a struck-out mono placeholder of the same glyph count resolves into the measured value,
 * so nothing reflows when it lands.
 */
export function TypographicNumberReveal({
  value,
  lower,
  upper,
  reveal,
  digits = 4,
}: {
  value: number;
  lower: number;
  upper: number;
  reveal: boolean;
  digits?: number;
}) {
  const reduced = useReducedMotion();
  const text = value.toFixed(digits);
  const placeholder = "0".repeat(text.length - 1).replace(/^./, "0.");
  const dur = reduced ? 0 : WT.base;

  return (
    <div className="font-mono">
      <div className="relative inline-block text-[3rem] leading-none sm:text-[3.6rem]">
        <motion.span
          className="text-ink-faint line-through decoration-hairline-strong decoration-2"
          initial={{ opacity: 1 }}
          animate={{ opacity: reveal ? 0 : 1 }}
          transition={{ duration: dur * 0.6, ease: WT.ease }}
          aria-hidden="true"
        >
          {placeholder}
        </motion.span>
        <motion.span
          className="absolute inset-0 text-ink"
          initial={{ opacity: 0, y: 6 }}
          animate={reveal ? { opacity: 1, y: 0 } : { opacity: 0, y: 6 }}
          transition={{ duration: dur, ease: WT.ease, delay: reduced ? 0 : dur * 0.4 }}
        >
          {text}
        </motion.span>
      </div>
      <motion.div
        className="mt-3 text-[13px] text-ink-soft"
        initial={{ opacity: 0 }}
        animate={{ opacity: reveal ? 1 : 0 }}
        transition={{ duration: dur, ease: WT.ease, delay: reduced ? 0 : dur * 0.7 }}
      >
        95% CI [{lower.toFixed(digits)}, {upper.toFixed(digits)}]
      </motion.div>
    </div>
  );
}
