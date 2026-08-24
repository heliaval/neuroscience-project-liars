import { useEffect, useState } from "react";
import { Navigate, Route, Routes, useLocation } from "react-router-dom";
import { SpotlightCursor } from "@/components/ui/spotlight-cursor";
import { Dashboard } from "@/routes/Dashboard";
import { Walkthrough } from "@/walkthrough/Walkthrough";

const GITHUB_REPO = "https://github.com/heliaval/neuroscience-project-liars";

/** A small centered utility row at the top of the viewport, plain words with no
 * background or border -- matches the floating nav text on motion.dev's site,
 * not a header bar. It hides itself on the walkthrough's cover screen so the
 * title screen opens clean; every other screen and /dashboard show it.
 *
 * The cover screen has no route of its own (the deck advances by state, not by
 * URL), so the signal is the footer's Back button: it is rendered on every
 * walkthrough screen but `disabled` only on the cover. Absent entirely means we
 * are not in the walkthrough at all (/dashboard) -- that case stays visible. */
function TopNav() {
  const location = useLocation();
  const [hidden, setHidden] = useState(false);

  useEffect(() => {
    const back = document.getElementById("wt-back-button") as HTMLButtonElement | null;
    // No Back button in the DOM => not the walkthrough (/dashboard) => show.
    if (!back) {
      setHidden(false);
      return;
    }

    const sync = () => setHidden(back.disabled);
    sync();

    // screenIndex changes without a route change, so this effect will not
    // re-run when the deck moves off the cover screen. The footer is rendered
    // outside Walkthrough's <AnimatePresence>, so the button node persists and
    // only its `disabled` attribute mutates -- watch that attribute directly.
    const observer = new MutationObserver(sync);
    observer.observe(back, { attributes: true, attributeFilter: ["disabled"] });
    return () => observer.disconnect();
  }, [location.pathname]);

  if (hidden) return null;

  return (
    <nav className="fixed top-4 left-1/2 z-50 flex -translate-x-1/2 gap-6 font-sans text-[11px] uppercase tracking-[0.1em] text-ink-faint">
      <a href={`${GITHUB_REPO}/issues`} target="_blank" rel="noreferrer" className="hover:text-ink">
        Report
      </a>
      <a href={GITHUB_REPO} target="_blank" rel="noreferrer" className="hover:text-ink">
        GitHub
      </a>
      <a href="https://www.youtube.com/watch?v=Scn449JjT2o" target="_blank" rel="noreferrer" className="hover:text-ink">
        Demo
      </a>
    </nav>
  );
}

function App() {
  return (
    <>
      <TopNav />
      <SpotlightCursor config={{ radius: 260, brightness: 0.05, color: "#ffffff" }} />
      <Routes>
        <Route path="/" element={<Walkthrough />} />
        <Route path="/dashboard" element={<Dashboard />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </>
  );
}

export default App;
