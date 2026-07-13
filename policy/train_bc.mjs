// Distill the MPPI expert into a tiny MLP (behavior cloning): [x,y,vx,vy] -> [ax,ay].
// Hand-rolled 4->32->32->2 ReLU net, Adam, MSE. Then validate the LEARNED policy
// closed-loop (does it capture like the expert?). Writes bc_policy.json for on-device use.
import fs from 'node:fs';

const D = JSON.parse(fs.readFileSync(new URL('./bc_data.json', import.meta.url)));
const rows = D.rows;
const N = rows.length, IN = 4, OUT = 2, H1 = 32, H2 = 32;

// --- normalize ---
const xs = rows.map(r => r.slice(0, 4)), ys = rows.map(r => r.slice(4, 6));
const mean = a => a.reduce((s, v) => s + v, 0) / a.length;
const std = (a, m) => Math.sqrt(a.reduce((s, v) => s + (v - m) ** 2, 0) / a.length) + 1e-8;
const xm = [], xsd = [], ym = [], ysd = [];
for (let j = 0; j < IN; j++) { const c = xs.map(r => r[j]); xm[j] = mean(c); xsd[j] = std(c, xm[j]); }
for (let j = 0; j < OUT; j++) { const c = ys.map(r => r[j]); ym[j] = mean(c); ysd[j] = std(c, ym[j]); }
const X = xs.map(r => r.map((v, j) => (v - xm[j]) / xsd[j]));
const Y = ys.map(r => r.map((v, j) => (v - ym[j]) / ysd[j]));

// --- params ---
let sd = 12345; const rnd = () => { sd = (sd * 1103515245 + 12345) & 0x7fffffff; return sd / 0x7fffffff; };
const randM = (r, c, s) => Array.from({ length: r }, () => Array.from({ length: c }, () => (rnd() * 2 - 1) * s));
const zeros = n => Array(n).fill(0);
let W1 = randM(IN, H1, Math.sqrt(2 / IN)), b1 = zeros(H1);
let W2 = randM(H1, H2, Math.sqrt(2 / H1)), b2 = zeros(H2);
let W3 = randM(H2, OUT, Math.sqrt(2 / H2)), b3 = zeros(OUT);
// Adam state
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
  let loss = 0, nb = 0;
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
    nb++;
  }
  if (ep % 10 === 9 || ep === 0) console.log(`epoch ${ep + 1}  train MSE ${(loss / N).toFixed(4)}`);
}

// --- forward (for validation + the shape shipped on-device) ---
function policy(sx) {
  const x = sx.map((v, j) => (v - xm[j]) / xsd[j]);
  const a1 = relu(matvec(W1, x).map((v, j) => v + b1[j]));
  const a2 = relu(matvec(W2, a1).map((v, j) => v + b2[j]));
  const yh = matvec(W3, a2).map((v, j) => v + b3[j]);
  return yh.map((v, j) => v * ysd[j] + ym[j]);   // denormalize to [ax, ay]
}

// --- validate: run the LEARNED policy closed-loop (double-integrator, ~= CW locally) ---
const DT = D.dt || 1.0; let caps = 0, T = 60;
for (let r = 0; r < T; r++) {
  let x = 16 + rnd() * 30, y = (rnd() * 2 - 1) * 13, vx = 0, vy = 0;
  for (let step = 0; step < 300; step++) {
    const u = policy([x, y, vx, vy]);
    vx += u[0] * DT; vy += u[1] * DT; x += vx * DT; y += vy * DT;
    if (Math.hypot(x, y) < 0.6 && Math.hypot(vx, vy) < 0.1) { caps++; break; }
  }
}
console.log(`LEARNED policy closed-loop: ${caps}/${T} captured`);

const params = { arch: [IN, H1, H2, OUT], act: 'relu', W1, b1, W2, b2, W3, b3, xm, xsd, ym, ysd };
fs.writeFileSync(new URL('./bc_policy.json', import.meta.url), JSON.stringify(params));
const nparams = IN*H1+H1 + H1*H2+H2 + H2*OUT+OUT;
console.log(`wrote bc_policy.json — ${nparams} parameters`);
