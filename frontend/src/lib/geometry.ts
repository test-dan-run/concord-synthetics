// Room geometry for the drawings. Room frame (artifacts/meeting.py): x along the length from the front wall
// (W, x = 0) to the back wall (E); y along the width from the S wall (y = 0) to the N wall; z up; angles in
// degrees counter-clockwise from +x. Drawings put the front wall on the left and N at the top, so screen y is
// flipped.

import type { Vec2, Vec3 } from "./api";

export function dist3(a: Vec3, b: Vec3): number {
  return Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
}

/** Wrap to (−180, 180]. */
export function wrapDeg(d: number): number {
  let x = ((d + 180) % 360 + 360) % 360 - 180;
  if (x === -180) x = 180;
  return x;
}

/** Direction from `a` to `b` in room degrees. */
export function angleTo(a: Vec2 | Vec3, b: Vec2 | Vec3): number {
  return (Math.atan2(b[1] - a[1], b[0] - a[0]) * 180) / Math.PI;
}

/** A seat's facing relative to the direction of the table centre: 0 faces the centre, + turns left. */
export function facingVsCentre(mouth: Vec3, facing: number, centre: Vec2): number {
  return wrapDeg(facing - angleTo(mouth, centre));
}

/** Room coordinates to drawing pixels: `scale` px per metre, `pad` px margin, N at the top. */
export function projector(roomWidth: number, scale: number, pad: number) {
  return {
    x: (x: number) => pad + x * scale,
    y: (y: number) => pad + (roomWidth - y) * scale,
    /** Room angle to an SVG direction vector (unit). */
    dir: (deg: number): Vec2 => [Math.cos((deg * Math.PI) / 180), -Math.sin((deg * Math.PI) / 180)],
  };
}

/** Seat radius in px: grows with the square root of turn share (UI doc, plan drawing conventions). */
export function seatRadius(share: number, base = 11, k = 26): number {
  return base + Math.sqrt(Math.max(0, share)) * k * 0.5;
}

export const speakerNumber = (id: string) => id.replace(/^S/, "");
