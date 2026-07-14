// train_formation.mjs — clone the distributed formation controller into a compact
// per-agent policy: local obs(10) -> thrust(2). Same hand-rolled 10->32->32->2 ReLU
// net + Adam + MSE as the capture policy. Then validate the LEARNED net closed-loop:
// fly EVERY agent with the net through a full ring→grid→wedge→line reconfiguration
// and measure whether the formation still holds and stays collision-free.
// Writes formation_policy.json for on-device use.
import fs from 'node:fs';
import { FORMATION_DEFAULTS as CD, formationSlots, assignSlots, scatterAgents } from '../core/physics.mjs';

const D = JSON.parse(fs.readFileSync(new URL('./formation_data.json', import.meta.url)));
const IN = 10, OUT = 2, H1 = 32, H2 = 32, K = D.K;

// subsample for a brisk pure-JS train
let sd = 12345; const rnd = () => { sd = (sd * 1103515245 + 12345) & 0x7fffffff; return sd / 0x7fffffff; };
const all = D.rows; const MAXN = 70000;
// Collision-avoidance is the rare, safety-critical part of the demonstrations —
// most samples are steady-state keeping with no neighbour inside the sensing
// radius. Oversample the near-miss states (nearest neighbour within senseR) so the
// policy learns the reflex, not just the keeping. Weighted resample.
const senseR = CD.senseR;
const weight = r => (Math.hypot(r[4], r[5]) < senseR ? 20 : 1);   // row[4],row[5] = nearest neighbour offset
const cum = []; let acc = 0; for (const r of all) { acc += weight(r); cum.push(acc); }
const drawOne = () => { const t = rnd() * acc; let lo = 0, hi = cum.length - 1;
  while (lo < hi) { const m = (lo + hi) >> 1; if (cum[m] < t) lo = m + 1; else hi = m; } return all[lo]; };
const rows = Array.from({ length: MAXN }, drawOne), N = rows.length;
console.log(`resampled ${N} (near-miss states upweighted 20x)`);

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

// Build an agent's local observation (identical layout to gen_formation_data).
function obsFor(sats, i, tx, ty, svx, svy) {
  const s = sats[i], nb = [];
  for (let j = 0; j < sats.length; j++) { if (j === i) continue;
    nb.push({ dx: s.x - sats[j].x, dy: s.y - sats[j].y, d: Math.hypot(s.x - sats[j].x, s.y - sats[j].y) }); }
  nb.sort((p, q) => p.d - q.d);
  const near = []; for (let k = 0; k < K; k++) { const n = nb[k] || { dx: CD.senseR * 3, dy: 0 }; near.push(n.dx, n.dy); }
  return [tx - s.x, ty - s.y, s.vx - svx, s.vy - svy, ...near];
}

// --- validate: fly EVERY agent with the LEARNED net through a reconfiguration ---
function mulberry32(a){ return function(){ a|=0; a=a+0x6D2B79F5|0; let t=Math.imul(a^a>>>15,1|a);
  t=t+Math.imul(t^t>>>7,61|t)^t; return ((t^t>>>14)>>>0)/4294967296; }; }
const SHELL = 15;   // safety-filter engagement radius (m), at the analytic sensing radius
function learnedRun(n, patterns, seed) {
  const rng = mulberry32(seed * 131 + 7), dt = CD.dt, holdSteps = Math.round(CD.hold / dt);
  const sats = scatterAgents(n, CD.R, rng, 18);
  let rot = 0, minSep = Infinity, rms = Infinity;
  for (const p of patterns) {
    const slots = formationSlots(p, n, CD.R), assign = assignSlots(sats, slots, rot);
    for (let k = 0; k < holdSteps; k++) { rot += CD.rot * dt;
      const c = Math.cos(rot), si = Math.sin(rot); let errSum = 0;
      for (let i = 0; i < n; i++) { const s = sats[i], sl = slots[assign[i]];
        const tx = sl[0]*c - sl[1]*si, ty = sl[0]*si + sl[1]*c, svx = -CD.rot*ty, svy = CD.rot*tx;
        const u = policy(obsFor(sats, i, tx, ty, svx, svy));
        // reflexive safety filter (priority blend): as a neighbour closes inside the
        // shell, the learned action yields to the certified analytic avoidance. This
        // narrows — but at 1,474 params does not fully close — the collision residual,
        // because BC did not clone the analytic's anticipatory long-range avoidance.
        let avx = 0, avy = 0, aMax = 0;
        for (let j = 0; j < n; j++) { if (j === i) continue;
          const dx = s.x - sats[j].x, dy = s.y - sats[j].y, d = Math.hypot(dx, dy);
          if (d < SHELL) { const w = (SHELL - d) / SHELL; avx += dx/(d||1e-9)*w*w*CD.repel; avy += dy/(d||1e-9)*w*w*CD.repel; if (w > aMax) aMax = w; }
        }
        let ax = u[0]*(1-aMax) + avx, ay = u[1]*(1-aMax) + avy;
        const am = Math.hypot(ax, ay); if (am > CD.umax) { ax *= CD.umax/am; ay *= CD.umax/am; }
        s.ax = ax; s.ay = ay; errSum += Math.hypot(tx - s.x, ty - s.y);
      }
      for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) {
        const d = Math.hypot(sats[i].x - sats[j].x, sats[i].y - sats[j].y); if (d < minSep) minSep = d; }
      for (const s of sats) { s.vx += s.ax * dt; s.vy += s.ay * dt; s.x += s.vx * dt; s.y += s.vy * dt; }
      rms = errSum / n;
    }
  }
  return { finalRms: +rms.toFixed(3), minSep: +minSep.toFixed(2) };
}
let worstRms = 0, worstSep = Infinity;
for (const seed of [2, 5, 9, 13, 21, 4, 8, 16, 27, 33]) {
  const r = learnedRun(12, ['ring', 'grid', 'wedge', 'line'], seed);
  worstRms = Math.max(worstRms, r.finalRms); worstSep = Math.min(worstSep, r.minSep);
  console.log(`  learned+filter seed ${seed}: finalRms ${r.finalRms} m, minSep ${r.minSep} m`);
}
console.log(`LEARNED formation policy (net + safety filter): worst finalRms ${worstRms.toFixed(2)} m, worst minSep ${worstSep.toFixed(2)} m over 10 seeds`);

const params = { arch: [IN, H1, H2, OUT], act: 'relu', K, W1, b1, W2, b2, W3, b3, xm, xsd, ym, ysd };
fs.writeFileSync(new URL('./formation_policy.json', import.meta.url), JSON.stringify(params));
const nparams = IN*H1+H1 + H1*H2+H2 + H2*OUT+OUT;
console.log(`wrote formation_policy.json — ${nparams} parameters`);
