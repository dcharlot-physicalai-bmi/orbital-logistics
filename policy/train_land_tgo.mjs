// train_land_tgo.mjs — the right factorisation for learned powered descent.
//
// Cloning the THRUST directly fails: the ZEM/ZEV gains go as 6/tgo², so the target is
// stiff exactly at touchdown, and no smooth net tracks it (plain BC lands 9/120; DAgger
// lifts that to 72/120 and plateaus, still short of the expert's 109/120).
//
// But the thrust is only stiff because of tgo. Split the law at its natural seam:
//   * the EXPENSIVE part is the time-to-go — the expert solves a scalar minimisation of
//     the closed-form energy cost at every single step. That is what is worth replacing.
//   * the EXACT part is ZEM/ZEV itself — closed form, and it is what gives the crisp
//     terminal behaviour. That is worth keeping.
// So the net learns tgo (a smooth scalar function of the state) and the closed form does
// the rest. The net replaces the optimisation, not the physics.
//
// state [dx, dy, vx, vy, g] -> tgo.  DAgger on top, since the net's own tgo errors still
// move the state distribution. Writes land_policy.json (mode: "tgo").
import fs from 'node:fs';
import { zemzev, optimalTgo, landFeedback } from '../core/physics.mjs';

const IN = 5, OUT = 1, H1 = 32, H2 = 32, dt = 0.1;
const BODIES = [1.62, 3.71, 9.81];
let sd = 12345; const rnd = () => { sd = (sd * 1103515245 + 12345) & 0x7fffffff; return sd / 0x7fffffff; };
function mulberry32(a){ return function(){ a|=0; a=a+0x6D2B79F5|0; let t=Math.imul(a^a>>>15,1|a);
  t=t+Math.imul(t^t>>>7,61|t)^t; return ((t^t>>>14)>>>0)/4294967296; }; }
function makeIC(seed, grav) {
  const rng = mulberry32(seed * 41 + 3);
  const twr = 1.8 + rng() * 1.9, aMax = twr * grav;
  const alt = 200 + rng() * 1000, lat = (rng() * 2 - 1) * 400;
  const vx = (rng() * 2 - 1) * 30, vy = -(10 + rng() * 60);
  return { r0: [lat, alt], v0: [vx, vy], grav, aMax };
}

// ---- seed demonstrations: the expert's own descents, labelled with its solved tgo ----
let DATA = [];
for (let seed = 1; seed <= 600; seed++) {
  const grav = BODIES[seed % BODIES.length], c = makeIC(seed, grav), g = [0, -grav];
  let r = c.r0.slice(), v = c.v0.slice();
  for (let k = 0; k < 4000; k++) {
    const tgo = optimalTgo(r, v, g);
    DATA.push([r[0], r[1], v[0], v[1], grav, tgo]);
    let a = zemzev(r, v, [0, 0], [0, 0], g, tgo);
    const am = Math.hypot(...a); if (am > c.aMax) a = a.map(x => x * c.aMax / am);
    v = [v[0] + a[0] * dt, v[1] + (a[1] - grav) * dt];
    r = [r[0] + v[0] * dt, r[1] + v[1] * dt];
    if (r[1] <= 0) break;
  }
}
console.log(`seed demonstrations: ${DATA.length} (state -> optimal time-to-go)`);

// ---- net ----
const relu = z => z.map(v => Math.max(0, v));
const matvec = (W, x) => W[0].map((_, j) => x.reduce((s, xi, i) => s + xi * W[i][j], 0));
const randM = (r, c, s) => Array.from({ length: r }, () => Array.from({ length: c }, () => (rnd() * 2 - 1) * s));
const zeros = n => Array(n).fill(0);
let W1 = randM(IN, H1, Math.sqrt(2 / IN)), b1 = zeros(H1);
let W2 = randM(H1, H2, Math.sqrt(2 / H1)), b2 = zeros(H2);
let W3 = randM(H2, OUT, Math.sqrt(2 / H2)), b3 = zeros(OUT);
let xm = [], xsd = [], ym = [], ysd = [];
const mk = m => m.map(r => r.map(() => 0)); const mkv = n => zeros(n);
let mW1 = mk(W1), vW1 = mk(W1), mb1 = mkv(H1), vb1 = mkv(H1);
let mW2 = mk(W2), vW2 = mk(W2), mb2 = mkv(H2), vb2 = mkv(H2);
let mW3 = mk(W3), vW3 = mk(W3), mb3 = mkv(OUT), vb3 = mkv(OUT);
const LR = 2e-3, B1 = 0.9, B2 = 0.999, EPS = 1e-8, BATCH = 256, MAXN = 70000;
let tstep = 0;
function adam(P, g, m, v, is2d) {
  tstep++; const bc1 = 1 - B1 ** tstep, bc2 = 1 - B2 ** tstep;
  if (is2d) for (let i = 0; i < P.length; i++) for (let j = 0; j < P[i].length; j++) {
    m[i][j] = B1 * m[i][j] + (1 - B1) * g[i][j]; v[i][j] = B2 * v[i][j] + (1 - B2) * g[i][j] ** 2;
    P[i][j] -= LR * (m[i][j] / bc1) / (Math.sqrt(v[i][j] / bc2) + EPS);
  } else for (let i = 0; i < P.length; i++) {
    m[i] = B1 * m[i] + (1 - B1) * g[i]; v[i] = B2 * v[i] + (1 - B2) * g[i] ** 2;
    P[i] -= LR * (m[i] / bc1) / (Math.sqrt(v[i] / bc2) + EPS);
  }
}
function predTgo(sx) {
  const x = sx.map((v, j) => (v - xm[j]) / xsd[j]);
  const a1 = relu(matvec(W1, x).map((v, j) => v + b1[j]));
  const a2 = relu(matvec(W2, a1).map((v, j) => v + b2[j]));
  const y = matvec(W3, a2).map((v, j) => v + b3[j])[0] * ysd[0] + ym[0];
  return Math.max(0.5, y);                       // a time-to-go is positive
}
function train(epochs) {
  const pool = DATA.length > MAXN ? Array.from({ length: MAXN }, () => DATA[Math.floor(rnd() * DATA.length)]) : DATA;
  const N = pool.length;
  const xsR = pool.map(r => r.slice(0, IN)), ysR = pool.map(r => r.slice(IN, IN + OUT));
  const mean = a => a.reduce((s, v) => s + v, 0) / a.length;
  const std = (a, m) => Math.sqrt(a.reduce((s, v) => s + (v - m) ** 2, 0) / a.length) + 1e-8;
  xm = []; xsd = []; ym = []; ysd = [];
  for (let j = 0; j < IN; j++) { const c = xsR.map(r => r[j]); xm[j] = mean(c); xsd[j] = std(c, xm[j]); }
  for (let j = 0; j < OUT; j++) { const c = ysR.map(r => r[j]); ym[j] = mean(c); ysd[j] = std(c, ym[j]); }
  const X = xsR.map(r => r.map((v, j) => (v - xm[j]) / xsd[j]));
  const Y = ysR.map(r => r.map((v, j) => (v - ym[j]) / ysd[j]));
  const idx = [...Array(N).keys()];
  let loss = 0;
  for (let ep = 0; ep < epochs; ep++) {
    for (let i = N - 1; i > 0; i--) { const j = Math.floor(rnd() * (i + 1)); [idx[i], idx[j]] = [idx[j], idx[i]]; }
    loss = 0;
    for (let b = 0; b < N; b += BATCH) {
      const batch = idx.slice(b, b + BATCH), B = batch.length;
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
  }
  return loss / N;
}
// fly with the LEARNED tgo + the exact closed form; optionally aggregate expert labels
function fly(c, collect) {
  let r = c.r0.slice(), v = c.v0.slice(); const g = [0, -c.grav];
  for (let k = 0; k < 4000; k++) {
    if (collect) collect.push([r[0], r[1], v[0], v[1], c.grav, optimalTgo(r, v, g)]);
    const tgo = predTgo([r[0], r[1], v[0], v[1], c.grav]);      // the net replaces the minimisation
    let a = zemzev(r, v, [0, 0], [0, 0], g, tgo);               // the closed form does the rest
    const am = Math.hypot(...a); if (am > c.aMax) a = a.map(x => x * c.aMax / am);
    v = [v[0] + a[0] * dt, v[1] + (a[1] - c.grav) * dt];
    r = [r[0] + v[0] * dt, r[1] + v[1] * dt];
    if (r[1] <= 0) return { landed: Math.abs(r[0]) < 8 && Math.hypot(...v) < 2.0, miss: Math.abs(r[0]), speed: Math.hypot(...v) };
  }
  return { landed: false, miss: Math.abs(r[0]), speed: Math.hypot(...v) };
}
function validate() {
  let ok = 0, tot = 0, ms = 0, sp = 0;
  for (const grav of BODIES) for (let seed = 5000; seed < 5040; seed++) {
    const r = fly(makeIC(seed, grav), null); tot++;
    if (r.landed) { ok++; ms += r.miss; sp += r.speed; }
  }
  return { ok, tot, miss: ok ? ms / ok : NaN, speed: ok ? sp / ok : NaN };
}

const ROUNDS = 4;
for (let round = 0; round <= ROUNDS; round++) {
  const mse = train(round === 0 ? 40 : 20);
  const v = validate();
  console.log(`round ${round}: |D|=${DATA.length}  MSE ${mse.toFixed(4)}  ->  ${v.ok}/${v.tot} soft` +
    (v.ok ? `  miss ${v.miss.toFixed(2)} m, touchdown ${v.speed.toFixed(2)} m/s` : ''));
  if (round === ROUNDS) break;
  const fresh = [];
  for (let seed = 1; seed <= 300; seed++) fly(makeIC(seed, BODIES[seed % BODIES.length]), fresh);
  DATA = DATA.concat(fresh);
}

const v = validate();
let expOk = 0;
for (const grav of BODIES) for (let seed = 5000; seed < 5040; seed++) {
  const c = makeIC(seed, grav);
  if (landFeedback(c.r0, c.v0, [0, -c.grav], c.aMax, { dt }).landed) expOk++;
}
console.log(`\nLEARNED tgo policy: ${v.ok}/${v.tot} soft landings on held-out starts (expert ${expOk}/${v.tot})` +
  (v.ok ? `, mean miss ${v.miss.toFixed(2)} m, mean touchdown ${v.speed.toFixed(2)} m/s` : ''));
const params = { mode: 'tgo', arch: [IN, H1, H2, OUT], act: 'relu', W1, b1, W2, b2, W3, b3, xm, xsd, ym, ysd };
fs.writeFileSync(new URL('./land_policy.json', import.meta.url), JSON.stringify(params));
console.log(`wrote land_policy.json — ${IN*H1+H1 + H1*H2+H2 + H2*OUT+OUT} parameters (predicts time-to-go)`);
