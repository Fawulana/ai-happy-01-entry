// ai-happy-01 entry (Node, standard library only): procedurally draws one
// image — a chubby orange tabby sitting blissfully in a golden sunbeam — and
// writes it to /output/entry.png. Every shape is drawn in code with signed
// distance functions and soft edges; no image is copied or downloaded, no
// network, no clock, and the only randomness is a fixed-seed PRNG for the
// floating dust motes, so two runs write identical bytes.
//
// Local run: mkdir -p out && MAKEFUN_OUTPUT_DIR=out node main.ts

import { deflateSync } from 'node:zlib';
import { mkdirSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const W = 1024;
const H = 1024;

// ---------------------------------------------------------------- utilities

const clamp = (v: number, a: number, b: number): number => (v < a ? a : v > b ? b : v);
const lerp = (a: number, b: number, t: number): number => a + (b - a) * t;
const smoothstep = (e0: number, e1: number, x: number): number => {
  const t = clamp((x - e0) / (e1 - e0), 0, 1);
  return t * t * (3 - 2 * t);
};

type RGB = [number, number, number];
const mixc = (a: RGB, b: RGB, t: number): RGB => [lerp(a[0], b[0], t), lerp(a[1], b[1], t), lerp(a[2], b[2], t)];

// Deterministic PRNG (mulberry32) for the dust motes.
function mulberry32(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// ---------------------------------------------------------------- framebuffer

// Float RGB buffer, 0..255 per channel.
const buf = new Float32Array(W * H * 3);

function blend(x: number, y: number, c: RGB, a: number): void {
  if (a <= 0 || x < 0 || y < 0 || x >= W || y >= H) return;
  if (a > 1) a = 1;
  const i = (y * W + x) * 3;
  const ia = 1 - a;
  buf[i] = buf[i] * ia + c[0] * a;
  buf[i + 1] = buf[i + 1] * ia + c[1] * a;
  buf[i + 2] = buf[i + 2] * ia + c[2] * a;
}

// Coverage of a pixel from a signed distance (negative inside) with a soft edge.
const cov = (d: number, feather: number): number => clamp(0.5 - d / (2 * feather), 0, 1);

// ---------------------------------------------------------------- SDF helpers

const ellipseD = (x: number, y: number, cx: number, cy: number, rx: number, ry: number): number => {
  const qx = (x - cx) / rx;
  const qy = (y - cy) / ry;
  return (Math.sqrt(qx * qx + qy * qy) - 1) * Math.min(rx, ry);
};

const circleD = (x: number, y: number, cx: number, cy: number, r: number): number =>
  Math.hypot(x - cx, y - cy) - r;

function segD(px: number, py: number, ax: number, ay: number, bx: number, by: number): number {
  const dx = bx - ax;
  const dy = by - ay;
  const l2 = dx * dx + dy * dy;
  const t = l2 === 0 ? 0 : clamp(((px - ax) * dx + (py - ay) * dy) / l2, 0, 1);
  return Math.hypot(px - (ax + dx * t), py - (ay + dy * t));
}

// Rounded triangle SDF (Inigo Quilez's sdTriangle, minus the rounding radius).
function triD(
  px: number, py: number,
  ax: number, ay: number, bx: number, by: number, cx: number, cy: number,
): number {
  const e0x = bx - ax, e0y = by - ay;
  const e1x = cx - bx, e1y = cy - by;
  const e2x = ax - cx, e2y = ay - cy;
  const v0x = px - ax, v0y = py - ay;
  const v1x = px - bx, v1y = py - by;
  const v2x = px - cx, v2y = py - cy;
  const t0 = clamp((v0x * e0x + v0y * e0y) / (e0x * e0x + e0y * e0y), 0, 1);
  const t1 = clamp((v1x * e1x + v1y * e1y) / (e1x * e1x + e1y * e1y), 0, 1);
  const t2 = clamp((v2x * e2x + v2y * e2y) / (e2x * e2x + e2y * e2y), 0, 1);
  const q0x = v0x - e0x * t0, q0y = v0y - e0y * t0;
  const q1x = v1x - e1x * t1, q1y = v1y - e1y * t1;
  const q2x = v2x - e2x * t2, q2y = v2y - e2y * t2;
  const s = Math.sign(e0x * e2y - e0y * e2x);
  const d0 = q0x * q0x + q0y * q0y;
  const d1 = q1x * q1x + q1y * q1y;
  const d2 = q2x * q2x + q2y * q2y;
  const c0 = s * (v0x * e0y - v0y * e0x);
  const c1 = s * (v1x * e1y - v1y * e1x);
  const c2 = s * (v2x * e2y - v2y * e2x);
  let m = d0;
  let mc = c0;
  if (d1 < m) { m = d1; mc = c1; }
  if (d2 < m) { m = d2; mc = c2; }
  return -Math.sqrt(m) * Math.sign(mc);
}

// ---------------------------------------------------------------- shape fills

function fillEllipse(cx: number, cy: number, rx: number, ry: number, c: RGB, alpha: number, feather = 1.6): void {
  const pad = feather + 2;
  const x0 = Math.max(0, Math.floor(cx - rx - pad));
  const x1 = Math.min(W - 1, Math.ceil(cx + rx + pad));
  const y0 = Math.max(0, Math.floor(cy - ry - pad));
  const y1 = Math.min(H - 1, Math.ceil(cy + ry + pad));
  for (let y = y0; y <= y1; y++) {
    for (let x = x0; x <= x1; x++) {
      const a = cov(ellipseD(x + 0.5, y + 0.5, cx, cy, rx, ry), feather) * alpha;
      blend(x, y, c, a);
    }
  }
}

function fillCircle(cx: number, cy: number, r: number, c: RGB, alpha: number, feather = 1.6): void {
  fillEllipse(cx, cy, r, r, c, alpha, feather);
}

function fillRotEllipse(
  cx: number, cy: number, rx: number, ry: number, ang: number, c: RGB, alpha: number, feather = 1.6,
): void {
  const cos = Math.cos(ang);
  const sin = Math.sin(ang);
  const R = Math.max(rx, ry) + feather + 2;
  const x0 = Math.max(0, Math.floor(cx - R));
  const x1 = Math.min(W - 1, Math.ceil(cx + R));
  const y0 = Math.max(0, Math.floor(cy - R));
  const y1 = Math.min(H - 1, Math.ceil(cy + R));
  for (let y = y0; y <= y1; y++) {
    for (let x = x0; x <= x1; x++) {
      const dx = x + 0.5 - cx;
      const dy = y + 0.5 - cy;
      const lx = dx * cos + dy * sin;
      const ly = -dx * sin + dy * cos;
      const a = cov(ellipseD(lx, ly, 0, 0, rx, ry), feather) * alpha;
      blend(x, y, c, a);
    }
  }
}

function fillTri(
  ax: number, ay: number, bx: number, by: number, cx: number, cy: number,
  round: number, c: RGB, alpha: number, feather = 1.6,
): void {
  const pad = feather + 2;
  const x0 = Math.max(0, Math.floor(Math.min(ax, bx, cx) - pad));
  const x1 = Math.min(W - 1, Math.ceil(Math.max(ax, bx, cx) + pad));
  const y0 = Math.max(0, Math.floor(Math.min(ay, by, cy) - pad));
  const y1 = Math.min(H - 1, Math.ceil(Math.max(ay, by, cy) + pad));
  for (let y = y0; y <= y1; y++) {
    for (let x = x0; x <= x1; x++) {
      const d = triD(x + 0.5, y + 0.5, ax, ay, bx, by, cx, cy) - round;
      blend(x, y, c, cov(d, feather) * alpha);
    }
  }
}

// Soft disc with a quadratic falloff (glows, blush, dust).
function softDisc(cx: number, cy: number, r: number, c: RGB, alpha: number): void {
  const pad = 2;
  const x0 = Math.max(0, Math.floor(cx - r - pad));
  const x1 = Math.min(W - 1, Math.ceil(cx + r + pad));
  const y0 = Math.max(0, Math.floor(cy - r - pad));
  const y1 = Math.min(H - 1, Math.ceil(cy + r + pad));
  for (let y = y0; y <= y1; y++) {
    for (let x = x0; x <= x1; x++) {
      const q = Math.hypot(x + 0.5 - cx, y + 0.5 - cy) / r;
      if (q >= 1) continue;
      const f = (1 - q * q) * (1 - q * q);
      blend(x, y, c, f * alpha);
    }
  }
}

// Stroke a polyline with round caps and a soft edge.
function strokePolyline(pts: Array<[number, number]>, width: number, c: RGB, alpha: number, feather = 1.4): void {
  let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
  for (const [px, py] of pts) {
    x0 = Math.min(x0, px); x1 = Math.max(x1, px);
    y0 = Math.min(y0, py); y1 = Math.max(y1, py);
  }
  const pad = width / 2 + feather + 2;
  x0 = Math.max(0, Math.floor(x0 - pad)); x1 = Math.min(W - 1, Math.ceil(x1 + pad));
  y0 = Math.max(0, Math.floor(y0 - pad)); y1 = Math.min(H - 1, Math.ceil(y1 + pad));
  for (let y = y0; y <= y1; y++) {
    for (let x = x0; x <= x1; x++) {
      let d = Infinity;
      for (let i = 0; i + 1 < pts.length; i++) {
        d = Math.min(d, segD(x + 0.5, y + 0.5, pts[i][0], pts[i][1], pts[i + 1][0], pts[i + 1][1]));
      }
      blend(x, y, c, cov(d - width / 2, feather) * alpha);
    }
  }
}

// Arc polyline points: circle centre, radius, angles in degrees (y-down).
function arcPts(cx: number, cy: number, r: number, a0: number, a1: number, n = 24): Array<[number, number]> {
  const pts: Array<[number, number]> = [];
  for (let i = 0; i <= n; i++) {
    const t = (a0 + ((a1 - a0) * i) / n) * (Math.PI / 180);
    pts.push([cx + r * Math.cos(t), cy + r * Math.sin(t)]);
  }
  return pts;
}

// ---------------------------------------------------------------- palette

const CREAM: RGB = [255, 243, 220];
const PEACH_TOP: RGB = [255, 231, 194];
const PEACH_MID: RGB = [255, 213, 158];
const PEACH_LOW: RGB = [255, 194, 138];
const SUN_CORE: RGB = [255, 248, 224];
const SUN_GLOW: RGB = [255, 232, 175];
const RAY: RGB = [255, 246, 214];
const WOOD_TOP: RGB = [216, 158, 104];
const WOOD_LOW: RGB = [178, 124, 76];
const WOOD_LINE: RGB = [150, 102, 58];
const ORANGE: RGB = [246, 169, 78];
const ORANGE_DEEP: RGB = [224, 139, 50];
const ORANGE_DARK: RGB = [198, 118, 40];
const BELLY: RGB = [255, 246, 226];
const PINK: RGB = [232, 126, 126];
const INNER_EAR: RGB = [247, 184, 168];
const BLUSH: RGB = [244, 154, 160];
const BROWN: RGB = [107, 74, 47];
const SHADOW: RGB = [138, 90, 48];

// ---------------------------------------------------------------- background

const SUN_X = 770;
const SUN_Y = 225;

for (let y = 0; y < H; y++) {
  const t = y / (H - 1);
  const base = t < 0.55 ? mixc(PEACH_TOP, PEACH_MID, t / 0.55) : mixc(PEACH_MID, PEACH_LOW, (t - 0.55) / 0.45);
  for (let x = 0; x < W; x++) {
    const i = (y * W + x) * 3;
    buf[i] = base[0]; buf[i + 1] = base[1]; buf[i + 2] = base[2];
  }
}

// Sun disc + halo.
softDisc(SUN_X, SUN_Y, 92, SUN_CORE, 0.95);
softDisc(SUN_X, SUN_Y, 260, SUN_GLOW, 0.6);
softDisc(SUN_X, SUN_Y, 520, SUN_GLOW, 0.38);

// Slanted light rays from the sun, drawn in rotated coordinates.
function drawRay(angDeg: number, halfW: number, len: number, alpha: number): void {
  const ang = (angDeg * Math.PI) / 180;
  const cos = Math.cos(ang);
  const sin = Math.sin(ang);
  const R = len + halfW + 4;
  const x0 = Math.max(0, Math.floor(SUN_X - R));
  const x1 = Math.min(W - 1, Math.ceil(SUN_X + R));
  const y0 = Math.max(0, Math.floor(SUN_Y - R));
  const y1 = Math.min(H - 1, Math.ceil(SUN_Y + R));
  for (let y = y0; y <= y1; y++) {
    for (let x = x0; x <= x1; x++) {
      const dx = x + 0.5 - SUN_X;
      const dy = y + 0.5 - SUN_Y;
      const u = dx * cos + dy * sin; // across the beam
      const v = -dx * sin + dy * cos; // along the beam
      if (v < 0 || v > len) continue;
      const across = 1 - smoothstep(halfW * 0.55, halfW, Math.abs(u));
      const along = 1 - smoothstep(len * 0.55, len, v);
      const near = smoothstep(60, 200, v);
      blend(x, y, RAY, across * along * near * alpha);
    }
  }
}
drawRay(118, 78, 900, 0.22);
drawRay(138, 52, 820, 0.18);
drawRay(100, 38, 760, 0.15);

// Soft halo behind the cat so it separates from the background.
softDisc(512, 520, 500, [255, 244, 214], 0.3);

// Wooden sill the cat sits on.
const SILL_Y = 858;
for (let y = SILL_Y; y < H; y++) {
  const t = (y - SILL_Y) / (H - SILL_Y);
  const c = mixc(WOOD_TOP, WOOD_LOW, t);
  // Soften the top edge over ~12px so the sill meets the wall gently.
  const edge = y < SILL_Y + 12 ? 0.15 + (0.85 * (y - SILL_Y)) / 12 : 1;
  for (let x = 0; x < W; x++) blend(x, y, c, edge);
}
// Sill top highlight + plank seams.
for (let x = 0; x < W; x++) {
  blend(x, SILL_Y, [238, 190, 138], 0.8);
  blend(x, SILL_Y + 1, [228, 176, 122], 0.5);
}
for (const py of [916, 968]) {
  for (let x = 0; x < W; x++) {
    blend(x, py, WOOD_LINE, 0.35);
    blend(x, py + 1, WOOD_LINE, 0.2);
  }
}
// Soft contact shadow under the cat.
{
  const cx = 512, cy = 884, rx = 305, ry = 50;
  for (let y = Math.floor(cy - ry - 4); y <= Math.ceil(cy + ry + 4); y++) {
    for (let x = Math.floor(cx - rx - 4); x <= Math.ceil(cx + rx + 4); x++) {
      const qx = (x + 0.5 - cx) / rx;
      const qy = (y + 0.5 - cy) / ry;
      const q = qx * qx + qy * qy;
      if (q >= 1) continue;
      blend(x, y, SHADOW, (1 - q) * 0.34);
    }
  }
}

// ---------------------------------------------------------------- the cat

const HEAD_CX = 512;
const HEAD_CY = 420;
const HEAD_R = 185;
const BODY_CX = 512;
const BODY_CY = 700;
const BODY_RX = 270;
const BODY_RY = 195;

// Ears (drawn first so the head overlaps their bases). Outline, then fill.
fillTri(352, 312, 414, 168, 502, 288, 14, ORANGE_DEEP, 1, 2);
fillTri(672, 312, 610, 168, 522, 288, 14, ORANGE_DEEP, 1, 2);
fillTri(362, 304, 416, 182, 492, 280, 12, ORANGE, 1, 2);
fillTri(662, 304, 608, 182, 532, 280, 12, ORANGE, 1, 2);
// Inner ears.
fillTri(388, 284, 424, 208, 472, 268, 10, INNER_EAR, 1, 2);
fillTri(636, 284, 600, 208, 552, 268, 10, INNER_EAR, 1, 2);

// Body: outline then fill.
fillEllipse(BODY_CX, BODY_CY, BODY_RX + 7, BODY_RY + 7, ORANGE_DEEP, 1, 2);
fillEllipse(BODY_CX, BODY_CY, BODY_RX, BODY_RY, ORANGE, 1, 2);
// Head: outline then fill.
fillCircle(HEAD_CX, HEAD_CY, HEAD_R + 7, ORANGE_DEEP, 1, 2);
fillCircle(HEAD_CX, HEAD_CY, HEAD_R, ORANGE, 1, 2);

// Cream chest patch.
fillEllipse(BODY_CX, 742, 150, 132, BELLY, 1, 2);

// Tail: resting on the sill to the cat's left, tip curled up.
{
  // Soft shadow under the tail.
  for (let y = 872; y <= 908; y++) {
    for (let x = 150; x <= 400; x++) {
      const qx = (x + 0.5 - 275) / 125;
      const qy = (y + 0.5 - 890) / 18;
      const q = qx * qx + qy * qy;
      if (q >= 1) continue;
      blend(x, y, SHADOW, (1 - q) * 0.2);
    }
  }
  const p0: [number, number] = [380, 845];
  const p1: [number, number] = [240, 905];
  const p2: [number, number] = [165, 848];
  const steps = 48;
  for (let i = 0; i <= steps; i++) {
    const t = i / steps;
    const mt = 1 - t;
    const x = mt * mt * p0[0] + 2 * mt * t * p1[0] + t * t * p2[0];
    const y = mt * mt * p0[1] + 2 * mt * t * p1[1] + t * t * p2[1];
    const r = lerp(38, 22, t);
    fillCircle(x, y, r + 5, ORANGE_DEEP, 1, 2);
  }
  for (let i = 0; i <= steps; i++) {
    const t = i / steps;
    const mt = 1 - t;
    const x = mt * mt * p0[0] + 2 * mt * t * p1[0] + t * t * p2[0];
    const y = mt * mt * p0[1] + 2 * mt * t * p1[1] + t * t * p2[1];
    const r = lerp(38, 22, t);
    fillCircle(x, y, r, ORANGE, 1, 2);
  }
  // Darker tail tip.
  fillCircle(p2[0], p2[1], 18, ORANGE_DEEP, 0.9, 2);
}

// Front paws tucked at the bottom.
fillEllipse(432, 852, 58, 42, ORANGE_DEEP, 1, 2);
fillEllipse(592, 852, 58, 42, ORANGE_DEEP, 1, 2);
fillEllipse(432, 850, 52, 36, ORANGE, 1, 2);
fillEllipse(592, 850, 52, 36, ORANGE, 1, 2);
strokePolyline([[432, 836], [432, 862]], 4, ORANGE_DARK, 0.6);
strokePolyline([[592, 836], [592, 862]], 4, ORANGE_DARK, 0.6);

// Tabby stripes: forehead + cheeks, soft and rounded.
strokePolyline([[512, 262], [512, 300]], 13, ORANGE_DARK, 0.55);
strokePolyline([[472, 270], [466, 306]], 12, ORANGE_DARK, 0.5);
strokePolyline([[552, 270], [558, 306]], 12, ORANGE_DARK, 0.5);
strokePolyline([[352, 380], [382, 392]], 11, ORANGE_DARK, 0.45);
strokePolyline([[348, 418], [380, 424]], 11, ORANGE_DARK, 0.45);
strokePolyline([[672, 380], [642, 392]], 11, ORANGE_DARK, 0.45);
strokePolyline([[676, 418], [644, 424]], 11, ORANGE_DARK, 0.45);
strokePolyline([[300, 640], [336, 648]], 12, ORANGE_DARK, 0.4);
strokePolyline([[724, 640], [688, 648]], 12, ORANGE_DARK, 0.4);

// Face: closed smiling eyes (upward arcs), nose, cat-smile mouth.
strokePolyline(arcPts(435, 440, 40, 195, 345), 8, BROWN, 0.95);
strokePolyline(arcPts(589, 440, 40, 195, 345), 8, BROWN, 0.95);
fillTri(494, 466, 530, 466, 512, 492, 7, PINK, 1, 1.6);
strokePolyline(arcPts(491, 494, 21, 25, 155), 6, BROWN, 0.9);
strokePolyline(arcPts(533, 494, 21, 25, 155), 6, BROWN, 0.9);

// Blush cheeks.
softDisc(388, 502, 52, BLUSH, 0.5);
softDisc(636, 502, 52, BLUSH, 0.5);

// Whiskers, three per side, gently fanned.
const whiskerL: Array<Array<[number, number]>> = [
  [[356, 462], [296, 440], [238, 428]],
  [[352, 486], [290, 482], [232, 482]],
  [[356, 510], [298, 522], [244, 540]],
];
const whiskerR: Array<Array<[number, number]>> = [
  [[668, 462], [728, 440], [786, 428]],
  [[672, 486], [734, 482], [792, 482]],
  [[668, 510], [726, 522], [780, 540]],
];
for (const w of [...whiskerL, ...whiskerR]) strokePolyline(w, 3, BROWN, 0.55);

// Warm rim light on the sun-facing edges of head and body, plus a soft
// shadow on the far side.
function rimLight(cx: number, cy: number, rx: number, ry: number): void {
  const sunDx = SUN_X - cx;
  const sunDy = SUN_Y - cy;
  const sunLen = Math.hypot(sunDx, sunDy);
  const sdx = sunDx / sunLen;
  const sdy = sunDy / sunLen;
  const pad = 10;
  const x0 = Math.max(0, Math.floor(cx - rx - pad));
  const x1 = Math.min(W - 1, Math.ceil(cx + rx + pad));
  const y0 = Math.max(0, Math.floor(cy - ry - pad));
  const y1 = Math.min(H - 1, Math.ceil(cy + ry + pad));
  for (let y = y0; y <= y1; y++) {
    for (let x = x0; x <= x1; x++) {
      const d = ellipseD(x + 0.5, y + 0.5, cx, cy, rx, ry);
      if (d >= 0) continue;
      const edge = clamp(1 + d / 46, 0, 1);
      if (edge <= 0) continue;
      const nx = (x + 0.5 - cx) / rx;
      const ny = (y + 0.5 - cy) / ry;
      const nl = Math.hypot(nx, ny) || 1;
      const facing = (nx / nl) * sdx + (ny / nl) * sdy;
      if (facing > 0) blend(x, y, [255, 241, 205], edge * facing * 0.28);
      else blend(x, y, ORANGE_DARK, edge * -facing * 0.1);
    }
  }
}
rimLight(HEAD_CX, HEAD_CY, HEAD_R, HEAD_R);
rimLight(BODY_CX, BODY_CY, BODY_RX, BODY_RY);

// ---------------------------------------------------------------- butterflies

function butterfly(cx: number, cy: number, s: number, tilt: number, c1: RGB, c2: RGB): void {
  const a = tilt;
  const cos = Math.cos(a);
  const sin = Math.sin(a);
  const put = (lx: number, ly: number): [number, number] => [
    cx + lx * cos - ly * sin,
    cy + lx * sin + ly * cos,
  ];
  const [w1x, w1y] = put(-16 * s, -10 * s);
  const [w2x, w2y] = put(16 * s, -10 * s);
  const [w3x, w3y] = put(-13 * s, 12 * s);
  const [w4x, w4y] = put(13 * s, 12 * s);
  fillRotEllipse(w1x, w1y, 20 * s, 13 * s, a - 0.5, c1, 1, 1.4);
  fillRotEllipse(w2x, w2y, 20 * s, 13 * s, a + 0.5, c1, 1, 1.4);
  fillRotEllipse(w3x, w3y, 13 * s, 9 * s, a - 0.4, c2, 1, 1.4);
  fillRotEllipse(w4x, w4y, 13 * s, 9 * s, a + 0.4, c2, 1, 1.4);
  const [b0x, b0y] = put(0, -12 * s);
  const [b1x, b1y] = put(0, 14 * s);
  strokePolyline([[b0x, b0y], [b1x, b1y]], 4 * s, BROWN, 0.9);
  const [a1x, a1y] = put(-7 * s, -20 * s);
  const [a2x, a2y] = put(7 * s, -20 * s);
  strokePolyline([[b0x, b0y], [a1x, a1y]], 2, BROWN, 0.7);
  strokePolyline([[b0x, b0y], [a2x, a2y]], 2, BROWN, 0.7);
}
butterfly(212, 318, 1.15, -0.35, [255, 158, 107], [255, 196, 107]);
butterfly(838, 462, 0.9, 0.3, [232, 160, 191], [242, 198, 216]);

// ---------------------------------------------------------------- flowers + heart

function flower(cx: number, cy: number, s: number, petal: RGB, centre: RGB): void {
  strokePolyline([[cx, cy + 6 * s], [cx, cy + 34 * s]], 4 * s, [122, 148, 84], 0.9);
  for (let i = 0; i < 5; i++) {
    const t = (i / 5) * Math.PI * 2 - Math.PI / 2;
    fillCircle(cx + Math.cos(t) * 11 * s, cy + Math.sin(t) * 11 * s, 8.5 * s, petal, 1, 1.4);
  }
  fillCircle(cx, cy, 7 * s, centre, 1, 1.4);
}
flower(866, 878, 1.25, [244, 140, 150], [255, 205, 92]);
flower(916, 894, 1.0, [255, 205, 92], [232, 126, 110]);
flower(820, 896, 0.95, [250, 250, 245], [255, 205, 92]);

// One small heart floating above the cat's left ear.
function heart(cx: number, cy: number, s: number, c: RGB, alpha: number): void {
  fillCircle(cx - 0.34 * s, cy - 0.22 * s, 0.38 * s, c, alpha, 1.6);
  fillCircle(cx + 0.34 * s, cy - 0.22 * s, 0.38 * s, c, alpha, 1.6);
  fillTri(cx - 0.64 * s, cy - 0.02 * s, cx + 0.64 * s, cy - 0.02 * s, cx, cy + 0.78 * s, 6, c, alpha, 1.6);
}
softDisc(262, 196, 52, [250, 170, 180], 0.35);
heart(262, 196, 44, [242, 126, 138], 1);

// ---------------------------------------------------------------- dust motes

{
  const rnd = mulberry32(0xc0ffee);
  for (let i = 0; i < 30; i++) {
    const x = 70 + 890 * Math.pow(rnd(), 0.75);
    const y = 70 + 640 * rnd();
    const r = 2.5 + rnd() * 5.5;
    softDisc(x, y, r, [255, 248, 225], 0.25 + rnd() * 0.35);
  }
}

// ---------------------------------------------------------------- vignette

for (let y = 0; y < H; y++) {
  for (let x = 0; x < W; x++) {
    const dx = (x + 0.5 - W / 2) / (W / 2);
    const dy = (y + 0.5 - H / 2) / (H / 2);
    const d = Math.sqrt(dx * dx + dy * dy);
    const v = smoothstep(0.72, 1.28, d) * 0.13;
    if (v > 0) blend(x, y, [166, 110, 60], v);
  }
}

// ---------------------------------------------------------------- PNG encoder

const CRC_TABLE = (() => {
  const table = new Int32Array(256);
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    table[n] = c;
  }
  return table;
})();

function crc32(data: Uint8Array): number {
  let c = 0xffffffff;
  for (let i = 0; i < data.length; i++) c = CRC_TABLE[(c ^ data[i]) & 0xff] ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
}

function chunk(type: string, data: Uint8Array): Uint8Array {
  const out = new Uint8Array(12 + data.length);
  const view = new DataView(out.buffer);
  view.setUint32(0, data.length);
  for (let i = 0; i < 4; i++) out[4 + i] = type.charCodeAt(i);
  out.set(data, 8);
  const crcData = new Uint8Array(4 + data.length);
  crcData.set(out.subarray(4, 8), 0);
  crcData.set(data, 4);
  view.setUint32(8 + data.length, crc32(crcData));
  return out;
}

function paeth(a: number, b: number, c: number): number {
  const p = a + b - c;
  const pa = Math.abs(p - a);
  const pb = Math.abs(p - b);
  const pc = Math.abs(p - c);
  if (pa <= pb && pa <= pc) return a;
  if (pb <= pc) return b;
  return c;
}

function encodePng(): Uint8Array {
  const raw = new Uint8Array(H * (1 + W * 3));
  const prev = new Uint8Array(W * 3);
  const cur = new Uint8Array(W * 3);
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const si = (y * W + x) * 3;
      cur[x * 3] = clamp(Math.round(buf[si]), 0, 255);
      cur[x * 3 + 1] = clamp(Math.round(buf[si + 1]), 0, 255);
      cur[x * 3 + 2] = clamp(Math.round(buf[si + 2]), 0, 255);
    }
    // Try filters None, Sub, Up, Paeth; keep the smallest sum of |signed| bytes.
    const candidates: Array<{ filter: number; line: Uint8Array; score: number }> = [];
    for (const filter of [0, 1, 2, 4]) {
      const line = new Uint8Array(W * 3);
      let score = 0;
      for (let i = 0; i < W * 3; i++) {
        const left = i >= 3 ? cur[i - 3] : 0;
        const up = prev[i];
        const upLeft = i >= 3 ? prev[i - 3] : 0;
        let pred = 0;
        if (filter === 1) pred = left;
        else if (filter === 2) pred = up;
        else if (filter === 4) pred = paeth(left, up, upLeft);
        const v = (cur[i] - pred) & 0xff;
        line[i] = v;
        score += v < 128 ? v : 256 - v;
      }
      candidates.push({ filter, line, score });
    }
    candidates.sort((a, b) => a.score - b.score);
    const best = candidates[0];
    raw[y * (1 + W * 3)] = best.filter;
    raw.set(best.line, y * (1 + W * 3) + 1);
    prev.set(cur);
  }
  const ihdr = new Uint8Array(13);
  const ihdrView = new DataView(ihdr.buffer);
  ihdrView.setUint32(0, W);
  ihdrView.setUint32(4, H);
  ihdr[8] = 8; // bit depth
  ihdr[9] = 2; // colour type RGB
  const idat = deflateSync(raw, { level: 9 });
  const parts = [
    Uint8Array.from([137, 80, 78, 71, 13, 10, 26, 10]),
    chunk('IHDR', ihdr),
    chunk('IDAT', idat),
    chunk('IEND', new Uint8Array(0)),
  ];
  const total = parts.reduce((n, p) => n + p.length, 0);
  const out = new Uint8Array(total);
  let off = 0;
  for (const p of parts) {
    out.set(p, off);
    off += p.length;
  }
  return out;
}

// ---------------------------------------------------------------- write output

const outDir = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';
mkdirSync(outDir, { recursive: true });
writeFileSync(join(outDir, 'entry.png'), encodePng());
