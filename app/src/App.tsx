import { lazy, Suspense, useEffect, useState } from "react";
import { Navigate, Route, Routes, useLocation } from "react-router-dom";
import { SpotlightCursor } from "@/components/ui/spotlight-cursor";
import { Walkthrough } from "@/walkthrough/Walkthrough";

// The dashboard is a secondary route -- most visits land on / (the walkthrough)
// and never touch it, so it's a separate chunk instead of weighing down the
// initial load every visitor pays for.
const Dashboard = lazy(() => import("@/routes/Dashboard").then((m) => ({ default: m.Dashboard })));

const GITHUB_REPO = "https://github.com/heliaval/neuroscience-project-liars";
const REPORT_DOC =
  "https://docs.google.com/document/d/14t2w2o-QyleDXb70hNdBxirVAnMybyngbeu_M6R5uuY/edit?tab=t.lr8mw0r8nie#heading=h.dplnxm61n8l7";

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
      <a href={REPORT_DOC} target="_blank" rel="noreferrer" className="hover:text-ink">
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
        <Route
          path="/dashboard"
          element={
            <Suspense fallback={null}>
              <Dashboard />
            </Suspense>
          }
        />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </>
  );
}

export default App;
