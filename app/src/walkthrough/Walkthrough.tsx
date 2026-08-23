import { useCallback, useEffect, useState, type MouseEvent } from "react";
import { CoverScreen } from "./screens/CoverScreen";
import { Exp1Screen } from "./screens/Exp1Screen";
import { Exp2Screen } from "./screens/Exp2Screen";
import { Exp3Screen } from "./screens/Exp3Screen";
import { Exp4Screen } from "./screens/Exp4Screen";
import { Exp5Screen } from "./screens/Exp5Screen";
import { Exp6Screen } from "./screens/Exp6Screen";
import { Exp7Screen } from "./screens/Exp7Screen";
import { ClosingScreen } from "./screens/ClosingScreen";
import { WalkthroughFooter } from "./WalkthroughFooter";

const SCREENS = [
  { id: "cover", maxStep: 0, render: CoverScreen },
  { id: "exp1", maxStep: 1, render: Exp1Screen },
  { id: "exp2", maxStep: 1, render: Exp2Screen },
  { id: "exp3", maxStep: 1, render: Exp3Screen },
  { id: "exp4", maxStep: 2, render: Exp4Screen },
  { id: "exp5", maxStep: 1, render: Exp5Screen },
  { id: "exp6", maxStep: 1, render: Exp6Screen },
  { id: "exp7", maxStep: 1, render: Exp7Screen },
  { id: "close", maxStep: 0, render: ClosingScreen },
] as const;

const EVIDENCE_COUNT = SCREENS.filter((s) => s.maxStep > 0).length; // 7

const SCREEN_TITLE: Record<(typeof SCREENS)[number]["id"], string> = {
  cover: "Overview",
  exp1: "Experiment 1",
  exp2: "Experiment 2",
  exp3: "Experiment 3",
  exp4: "Experiment 4",
  exp5: "Experiment 5",
  exp6: "Experiment 6",
  exp7: "Experiment 7",
  close: "Conclusion",
};

/** exp4 is the only screen with two reveal beats; its second click has its own label
 * so the control communicates a distinct action instead of repeating "Show the
 * evidence" for a step that has already shown evidence. */
function getAdvanceLabel(screenId: string, revealStep: number, maxStep: number, atLast: boolean): string {
  if (revealStep < maxStep) {
    if (screenId === "exp4" && revealStep === 1) return "See the same result in levels";
    return "Show the evidence";
  }
  return atLast ? "End" : "Next";
}

export function Walkthrough() {
  const [screenIndex, setScreenIndex] = useState(0);
  const [revealStep, setRevealStep] = useState(0);

  const screen = SCREENS[screenIndex];
  const atLast = screenIndex === SCREENS.length - 1;
  const canAdvance = revealStep < screen.maxStep || !atLast;
  const canBack = screenIndex > 0;

  const advance = useCallback(() => {
    setRevealStep((step) => {
      if (step < SCREENS[screenIndex].maxStep) return step + 1;
      if (screenIndex < SCREENS.length - 1) {
        setScreenIndex(screenIndex + 1);
        return 0;
      }
      return step;
    });
  }, [screenIndex]);

  // Back always lands on an UNrevealed screen, so re-entering a screen replays its reveal.
  const back = useCallback(() => {
    setScreenIndex((i) => Math.max(0, i - 1));
    setRevealStep(0);
  }, []);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      // A focused control (Back/Next, or TechnicalDetails' disclosure button) handles
      // its own Enter/Space activation natively -- intercepting here would silently
      // override that with "advance the deck" instead of "activate the focused control".
      if ((e.target as HTMLElement)?.closest?.("button, a, input, textarea, select")) return;
      if (["ArrowRight", "ArrowDown", " ", "Enter"].includes(e.key)) {
        e.preventDefault();
        advance();
      } else if (["ArrowLeft", "ArrowUp"].includes(e.key)) {
        e.preventDefault();
        back();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [advance, back]);

  // A click on the stage advances, but never when it landed on a control
  // (TechnicalDetails' disclosure, the dashboard link, the footer buttons), and never
  // when the click is the end of a text-selection drag -- otherwise selecting a
  // statistic to copy it silently discards the selection and advances the screen.
  const onStageClick = (e: MouseEvent<HTMLElement>) => {
    if ((e.target as HTMLElement).closest("button, a")) return;
    const selection = window.getSelection();
    if (selection && selection.toString().length > 0) return;
    advance();
  };

  useEffect(() => {
    document.title = `${SCREEN_TITLE[screen.id]} · Can your brain learn a liar`;
  }, [screen.id]);

  const Screen = screen.render;
  const evidenceIndex = screen.maxStep > 0 ? screenIndex : null;
  const revealed = revealStep >= screen.maxStep && screen.maxStep > 0;
  const advanceLabel = getAdvanceLabel(screen.id, revealStep, screen.maxStep, atLast);

  return (
    <div className="walkthrough flex min-h-[100dvh] flex-col bg-paper">
      <main className="flex flex-1 items-center" onClick={onStageClick}>
        <Screen revealStep={revealStep} />
      </main>
      <div aria-live="polite" className="sr-only">
        {revealed ? "Evidence revealed." : ""}
      </div>
      <WalkthroughFooter
        screenIndex={screenIndex}
        screenCount={SCREENS.length}
        evidenceIndex={evidenceIndex}
        evidenceCount={EVIDENCE_COUNT}
        canBack={canBack}
        canAdvance={canAdvance}
        onBack={back}
        onAdvance={advance}
        advanceLabel={advanceLabel}
      />
    </div>
  );
}
