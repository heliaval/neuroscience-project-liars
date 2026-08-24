/** Sizes a viewBox'd chart SVG.
 *
 * `fit` mode (the walkthrough deck): emits a CSS custom property holding the largest
 * aspect-locked width that fits BOTH the chart container's real width and its real
 * height, using container query units against the nearest `container-type: size`
 * ancestor. No viewport fraction is involved, so it is correct at every viewport
 * aspect ratio instead of at the one it was hand-tuned for.
 *
 * Non-fit mode (the /dashboard sections) keeps the plain full-width behaviour: those
 * call sites have no bounded-height ancestor, and container query units with no
 * eligible container would silently fall back to the viewport. */
export function chartBox(width: number, height: number, fit?: boolean): Record<string, string> {
  const style: Record<string, string> = { aspectRatio: `${width} / ${height}` };
  if (fit) style["--chart-fit-w"] = `min(100cqw, calc(100cqh * ${(width / height).toFixed(4)}))`;
  return style;
}
