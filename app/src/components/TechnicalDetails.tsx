import { useState } from "react";

export function TechnicalDetails({ text }: { text: string }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="mt-3">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="font-sans text-[13px] text-signal underline decoration-signal-soft decoration-2 underline-offset-4 hover:decoration-signal"
      >
        {open ? "Hide technical details" : "See technical details"}
      </button>
      {open && (
        <p className="mt-2 border-l-2 border-hairline-strong pl-4 font-mono text-[12px] leading-relaxed text-ink-soft">
          {text}
        </p>
      )}
    </div>
  );
}
