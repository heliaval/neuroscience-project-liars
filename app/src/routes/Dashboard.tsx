import { ProvenanceBanner } from "@/components/ProvenanceBanner";
import { Opening } from "@/sections/Opening";
import { StrangerVsFamiliar } from "@/sections/StrangerVsFamiliar";
import { PerDyadPanel } from "@/sections/PerDyadPanel";

/** The original dashboard page, moved from App.tsx unchanged when the walkthrough
 * took over the root route. Content and styling are deliberately identical to what
 * shipped before -- including the page-level ProvenanceBanner, which this page has
 * always rendered and must keep rendering. */
export function Dashboard() {
  return (
    <div className="min-h-[100dvh] bg-paper">
      <ProvenanceBanner />
      <main>
        <Opening />
        <div className="mx-auto max-w-[68ch] border-t border-hairline px-6 sm:px-8" />
        <StrangerVsFamiliar />
        <div className="mx-auto max-w-[68ch] border-t border-hairline px-6 sm:px-8" />
        <PerDyadPanel />
      </main>
    </div>
  );
}
