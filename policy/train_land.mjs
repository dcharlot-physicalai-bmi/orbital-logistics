// train_land.mjs — clone ZEM/ZEV powered-descent guidance into a compact reactive net:
// [dx, dy, vx, vy, g, aMax] -> [ax, ay]. Same hand-rolled 6->32->32->2 ReLU net + Adam
// + MSE as the capture and formation policies. Then validate the LEARNED net closed-loop
// on HELD-OUT initial conditions across Moon / Mars / Earth — flown as pure state
// feedback with NO time-to-go clock, which is the whole point: the analytic expert needs
// a flight-time schedule, the net does not. Writes land_policy.json for on-device use.
import fs from 'node:fs';
import { landFeedback } from '../core/physics.mjs';

const D = JSON.parse(fs.readFileSync(new URL('./land_data.json', import.meta.url)));
const IN = 6, OUT = 2, H1 = 32, H2 = 32;

let sd = 12345; const rnd = () => { sd = (sd * 1103515245 + 12345) & 0x7fffffff; return sd / 0x7fffffff; };
const all = D.rows, MAXN = 70000;
const rows = all.length > MAXN ? Array.from({ length: MAXN }, () => all[Math.floor(rnd() * all.length)]) : all;
const N = rows.length;
console.log(`training on ${N} of ${all.length} demonstrations`);

// --- normalize ---
const xs = rows.map(r => r.slice(0, IN)), ys = rows.map(r => r.slice(IN, IN + OUT));
const mean = a => a.reduce((s, v) => s + v, 0) / a.length;
const std = (a, m) => Math.sqrt(a.reduce((s, v) => s + (v - m) ** 2, 0) / a.length) + 1e-8;
const xm = [], xsd = [], ym = [], ysd = [];
for (let j = 0; j < IN; j++) { const c = xs.map(r => r[j]); xm[j] = mean(c); xsd[j] = std(c, xm[j]); }
for (let j = 0; j < OUT; j++) { const c = ys.map(r => r[j]); ym[j] = mean(c); ysd[j] = std(c, ym[j]); }
const X = xs.map(r => r.map((v, j) => (v - xm[j]) / xsd[j]));
const Y = ys.map(r => r.map((v, j) => (v - ym[j]) / ysd[j]));

// --- params ---
const randM = (r, c, s) => Array.from({ length: r }, () => Array.from({ length: c }, () => (rnd() * 2 - 1) * s));
const zeros = n => Array(n).fill(0);
let W1 = randM(IN, H1, Math.sqrt(2 / IN)), b1 = zeros(H1);
let W2 = randM(H1, H2, Math.sqrt(2 / H1)), b2 = zeros(H2);
let W3 = randM(H2, OUT, Math.sqrt(2 / H2)), b3 = zeros(OUT);
const mk = m => m.map(r => r.map(() => 0)); const mkv = n => zeros(n);
let mW1 = mk(W1), vW1 = mk(W1), mb1 = mkv(H1), vb1 = mkv(H1);
let mW2 = mk(W2), vW2 = mk(W2), mb2 = mkv(H2), vb2 = mkv(H2);
let mW3 = mk(W3), vW3 = mk(W3), mb3 = mkv(OUT), vb3 = mkv(OUT);
const relu = z => z.map(v => Math.max(0, v));
const matvec = (W, x) => W[0].map((_, j) => x.reduce((s, xi, i) => s + xi * W[i][j], 0));

const LR = 2e-3, B1 = 0.9, B2 = 0.999, EP = 1e-8, EPOCHS = 60, BATCH = 256;
let t = 0;
function adam(P, g, m, v, is2d) {
  t++; const bc1 = 1 - B1 ** t, bc2 = 1 - B2 ** t;
  if (is2d) for (let i = 0; i < P.length; i++) for (let j = 0; j < P[i].length; j++) {
    m[i][j] = B1 * m[i][j] + (1 - B1) * g[i][j]; v[i][j] = B2 * v[i][j] + (1 - B2) * g[i][j] ** 2;
    P[i][j] -= LR * (m[i][j] / bc1) / (Math.sqrt(v[i][j] / bc2) + EP);
  } else for (let i = 0; i < P.length; i++) {
    m[i] = B1 * m[i] + (1 - B1) * g[i]; v[i] = B2 * v[i] + (1 - B2) * g[i] ** 2;
    P[i] -= LR * (m[i] / bc1) / (Math.sqrt(v[i] / bc2) + EP);
  }
}
const idx = [...Array(N).keys()];
for (let ep = 0; ep < EPOCHS; ep++) {
  for (let i = N - 1; i > 0; i--) { const j = Math.floor(rnd() * (i + 1)); [idx[i], idx[j]] = [idx[j], idx[i]]; }
  let loss = 0;
  for (let b = 0; b < N; b += BATCH) {
    const batch = idx.slice(b, b + BATCH); const B = batch.length;
    const gW1 = mk(W1), gb1 = mkv(H1), gW2 = mk(W2), gb2 = mkv(H2), gW3 = mk(W3), gb3 = mkv(OUT);
    for (const s of batch) {
      const x = X[s], y = Y[s];
      const z1 = matvec(W1, x).map((v, j) => v + b1[j]), a1 = relu(z1);
      const z2 = matvec(W2, a1).map((v, j) => v + b2[j]), a2 = relu(z2);
      const yh = matvec(W3, a2).map((v, j) => v + b3[j]);
      const dy = yh.map((v, j) => 2 * (v - y[j]) / B); loss += yh.reduce((s, v, j) => s + (v - y[j]) ** 2, 0);
      for (let i2 = 0; i2 < H2; i2++) for (let j = 0; j < OUT; j++) gW3[i2][j] += a2[i2] * dy[j];
      for (let j = 0; j < OUT; j++) gb3[j] += dy[j];
      const da2 = a2.map((_, i2) => dy.reduce((s, dv, j) => s + dv * W3[i2][j], 0));
      const dz2 = da2.map((v, i2) => z2[i2] > 0 ? v : 0);
      for (let i1 = 0; i1 < H1; i1++) for (let j = 0; j < H2; j++) gW2[i1][j] += a1[i1] * dz2[j];
      for (let j = 0; j < H2; j++) gb2[j] += dz2[j];
      const da1 = a1.map((_, i1) => dz2.reduce((s, dv, j) => s + dv * W2[i1][j], 0));
      const dz1 = da1.map((v, i1) => z1[i1] > 0 ? v : 0);
      for (let i0 = 0; i0 < IN; i0++) for (let j = 0; j < H1; j++) gW1[i0][j] += x[i0] * dz1[j];
      for (let j = 0; j < H1; j++) gb1[j] += dz1[j];
    }
    adam(W3, gW3, mW3, vW3, true); adam(b3, gb3, mb3, vb3, false);
    adam(W2, gW2, mW2, vW2, true); adam(b2, gb2, mb2, vb2, false);
    adam(W1, gW1, mW1, vW1, true); adam(b1, gb1, mb1, vb1, false);
  }
  if (ep % 10 === 9 || ep === 0) console.log(`epoch ${ep + 1}  train MSE ${(loss / N).toFixed(4)}`);
}

// --- forward (validation + the shape shipped on-device) ---
function policy(sx) {
  const x = sx.map((v, j) => (v - xm[j]) / xsd[j]);
  const a1 = relu(matvec(W1, x).map((v, j) => v + b1[j]));
  const a2 = relu(matvec(W2, a1).map((v, j) => v + b2[j]));
  const yh = matvec(W3, a2).map((v, j) => v + b3[j]);
  return yh.map((v, j) => v * ysd[j] + ym[j]);
}

// --- validate: fly the LEARNED net closed-loop, pure state feedback, no clock ---
function mulberry32(a){ return function(){ a|=0; a=a+0x6D2B79F5|0; let t=Math.imul(a^a>>>15,1|a);
  t=t+Math.imul(t^t>>>7,61|t)^t; return ((t^t>>>14)>>>0)/4294967296; }; }
function flyLearned(r0, v0, grav, aMax, dt = 0.1) {
  let r = r0.slice(), v = v0.slice(), fuel = 0;
  for (let k = 0; k < 4000; k++) {
    let a = policy([r[0], r[1], v[0], v[1], grav, aMax]);
    const am = Math.hypot(...a); if (am > aMax) a = a.map(x => x * aMax / am);
    v = [v[0] + a[0] * dt, v[1] + (a[1] - grav) * dt];
    r = [r[0] + v[0] * dt, r[1] + v[1] * dt];
    fuel += Math.hypot(...a) * dt;
    if (r[1] <= 0) return { landed: Math.abs(r[0]) < 8 && Math.hypot(...v) < 2.0, miss: Math.abs(r[0]), speed: Math.hypot(...v), fuel };
  }
  return { landed: false, miss: Math.abs(r[0]), speed: Math.hypot(...v), fuel };
}
// The expert on the SAME held-out starts, so the comparison is like-for-like: the net
// can only be expected to land where the clock-free law it was cloned from can land.
const BODIES = [[1.62, 'Moon'], [3.71, 'Mars'], [9.81, 'Earth']];
let ok = 0, tot = 0, missSum = 0, spdSum = 0, expOk = 0;
for (const [grav, name] of BODIES) {
  let bok = 0, btot = 0, bmiss = 0, bspd = 0, bexp = 0;
  for (let seed = 5000; seed < 5040; seed++) {                  // HELD OUT (train used 1..900)
    const rng = mulberry32(seed * 41 + 3);
    const twr = 1.8 + rng() * 1.9, aMax = twr * grav;
    const alt = 200 + rng() * 1000, lat = (rng() * 2 - 1) * 400;
    const vx = (rng() * 2 - 1) * 30, vy = -(10 + rng() * 60);
    const e = landFeedback([lat, alt], [vx, vy], [0, -grav], aMax, { dt: 0.1 });
    if (e.landed) bexp++;
    const r = flyLearned([lat, alt], [vx, vy], grav, aMax);
    btot++; if (r.landed) { bok++; bmiss += r.miss; bspd += r.speed; }
  }
  ok += bok; tot += btot; missSum += bmiss; spdSum += bspd; expOk += bexp;
  console.log(`  ${name.padEnd(6)} expert ${bexp}/${btot} · learned ${bok}/${btot}` +
    (bok ? `  mean miss ${(bmiss/bok).toFixed(2)} m, touchdown ${(bspd/bok).toFixed(2)} m/s` : ''));
}
console.log(`LEARNED landing policy: ${ok}/${tot} soft landings on held-out starts (expert ${expOk}/${tot})` +
  (ok ? `, mean miss ${(missSum/ok).toFixed(2)} m, mean touchdown ${(spdSum/ok).toFixed(2)} m/s` : ''));

const params = { arch: [IN, H1, H2, OUT], act: 'relu', W1, b1, W2, b2, W3, b3, xm, xsd, ym, ysd };
fs.writeFileSync(new URL('./land_policy_thrust_bc.json', import.meta.url), JSON.stringify(params));
console.log(`wrote land_policy_thrust_bc.json — ${IN*H1+H1 + H1*H2+H2 + H2*OUT+OUT} parameters`);
