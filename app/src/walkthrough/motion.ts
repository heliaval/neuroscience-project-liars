/**
 * Walkthrough motion language. Deliberately louder than the rest of the app:
 * globals.css's "near-static motion" rule governs /dashboard and stays in force
 * there. This is the carve-out, and it is scoped -- nothing here reaches the dashboard.
 *
 * KEEP IN SYNC with the `.walkthrough` block in app/src/styles/globals.css.
 * These duplicate the CSS custom properties because motion's tween config takes
 * numbers, not CSS strings.
 */
export const WT = {
  /** seconds */
  fast: 0.26,
  base: 0.52,
  slow: 0.9,
  stagger: 0.06,
  /** slower stagger for the emphasised exp4 screen */
  staggerSlow: 0.1,
  /** tighter stagger for many-element figures (the 200-bar histogram) */
  staggerTight: 0.004,
  /** cubic-bezier(0.16, 1, 0.3, 1) -- a long decelerating settle */
  ease: [0.16, 1, 0.3, 1] as const,
  /** overshoot-and-settle spring, for figures that should feel kinetic (screen 6) */
  spring: { type: "spring", stiffness: 320, damping: 18, mass: 0.9 } as const,
} as const;

export const stageVariants = {
  hidden: {},
  shown: (staggerSeconds: number = WT.stagger) => ({
    transition: { staggerChildren: staggerSeconds, delayChildren: WT.fast },
  }),
};

export const itemVariants = {
  hidden: { opacity: 0, y: 12 },
  shown: { opacity: 1, y: 0, transition: { duration: WT.base, ease: WT.ease } },
};
