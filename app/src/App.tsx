import { ProvenanceBanner } from "@/components/ProvenanceBanner";
import { Opening } from "@/sections/Opening";
import { StrangerVsFamiliar } from "@/sections/StrangerVsFamiliar";
import { PerDyadPanel } from "@/sections/PerDyadPanel";

function App() {
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

export default App;
