// Inline stroke icons (24 px grid, currentColor). Decorative: the text next to them carries the meaning.

import type { SVGProps } from "react";

type P = SVGProps<SVGSVGElement> & { size?: number };

function Svg({ size = 16, children, ...p }: P) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={1.8}
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden
      {...p}
    >
      {children}
    </svg>
  );
}

/** The Synthetics mark (assets/icon.svg): a level meter rising to a spark. */
export const SyntheticsIcon = (p: P) => (
  <Svg {...p}>
    <path d="M3 11v6M7 7v14M11 10v8M15 12v4M19 13v2" />
    <path d="M19 1c.4 1.9 1.6 3.1 3.5 3.5-1.9.4-3.1 1.6-3.5 3.5-.4-1.9-1.6-3.1-3.5-3.5 1.9-.4 3.1-1.6 3.5-3.5z" />
  </Svg>
);
