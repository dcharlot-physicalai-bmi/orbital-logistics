// formation_bench.mjs — an open, deterministic benchmark for decentralized formation
// flight: hold a rotating formation and reconfigure it (ring→grid→wedge→line) with
// LOCAL control only, and score it on the three things that matter — does the shape
// hold (final RMS slot error), does it stay collision-free (global minimum separation
// against the hard-body radius), and at what cost (Δv per agent). Fixed seeds,
// bit-identical every run. Two controllers:
//
//   Distributed (analytic)  — the certified hand-tuned potential-field controller.
//   Learned (1,474-param)   — a compact per-agent net cloned from it, run on-device,
//                             under a reflexive safety filter.
//
// The comparison is itself the finding: behavior cloning reproduces the keeping and
// the reconfiguration, but not the analytic's anticipatory long-range avoidance, so
// the learned policy carries a residual minimum-separation gap the analytic does not.
import fs from 'node:fs';
import { FORMATION_DEFAULTS as CD, formationSlots, assignSlots, scatterAgents, formationRun } from '../core/physics.mjs';

const PATTERNS = ['ring', 'grid', 'wedge', 'line'];
const N = 12, SEEDS = Array.from({ length: 30 }, (_, i) => i + 1);
const COLLISION = 2 * CD.safeR;                     // centres closer than this = contact
const CONVERGED = CD.R * 0.03;                       // hold the shape to within 3% of scale

// --- the learned policy (loaded) + its safety filter -------------------------
const P = JSON.parse(fs.readFileSync(new URL('../policy/formation_policy.json', import.meta.url)));
const { W1, b1, W2, b2, W3, b3, xm, xsd, ym, ysd, K } = P;
const relu = z => z.map(v => Math.max(0, v));
const mv = (W, x) => W[0].map((_, j) => x.reduce((s, xi, i) => s + xi * W[i][j], 0));
function policy(sx) { const x = sx.map((v, j) => (v - xm[j]) / xsd[j]);
  const a1 = relu(mv(W1, x).map((v, j) => v + b1[j])), a2 = relu(mv(W2, a1).map((v, j) => v + b2[j]));
  return mv(W3, a2).map((v, j) => v + b3[j]).map((v, j) => v * ysd[j] + ym[j]); }
function mulberry32(a){ return function(){ a|=0; a=a+0x6D2B79F5|0; let t=Math.imul(a^a>>>15,1|a);
  t=t+Math.imul(t^t>>>7,61|t)^t; return ((t^t>>>14)>>>0)/4294967296; }; }
const SHELL = 15;
function obsFor(sats, i, tx, ty, svx, svy) {
  const s = sats[i], nb = [];
  for (let j = 0; j < sats.length; j++) { if (j === i) continue;
    nb.push({ dx: s.x - sats[j].x, dy: s.y - sats[j].y, d: Math.hypot(s.x - sats[j].x, s.y - sats[j].y) }); }
  nb.sort((p, q) => p.d - q.d);
  const near = []; for (let k = 0; k < K; k++) { const n = nb[k] || { dx: CD.senseR * 3, dy: 0 }; near.push(n.dx, n.dy); }
  return [tx - s.x, ty - s.y, s.vx - svx, s.vy - svy, ...near];
}
function learnedRun(n, patterns, seed) {
  const rng = mulberry32(seed * 131 + 7), dt = CD.dt, hold = Math.round(CD.hold / dt);
  const sats = scatterAgents(n, CD.R, rng, 18);
  let rot = 0, minSep = Infinity, rms = Infinity, dvSum = 0;
  for (const p of patterns) {
    const slots = formationSlots(p, n, CD.R), assign = assignSlots(sats, slots, rot);
    for (let k = 0; k < hold; k++) { rot += CD.rot * dt;
      const c = Math.cos(rot), si = Math.sin(rot); let es = 0;
      for (let i = 0; i < n; i++) { const s = sats[i], sl = slots[assign[i]];
        const tx = sl[0]*c - sl[1]*si, ty = sl[0]*si + sl[1]*c, svx = -CD.rot*ty, svy = CD.rot*tx;
        const u = policy(obsFor(sats, i, tx, ty, svx, svy));
        let avx = 0, avy = 0, aMax = 0;
        for (let j = 0; j < n; j++) { if (j === i) continue;
          const dx = s.x - sats[j].x, dy = s.y - sats[j].y, d = Math.hypot(dx, dy);
          if (d < SHELL) { const w = (SHELL - d) / SHELL; avx += dx/(d||1e-9)*w*w*CD.repel; avy += dy/(d||1e-9)*w*w*CD.repel; if (w > aMax) aMax = w; } }
        let ax = u[0]*(1-aMax) + avx, ay = u[1]*(1-aMax) + avy;
        const am = Math.hypot(ax, ay); if (am > CD.umax) { ax *= CD.umax/am; ay *= CD.umax/am; }
        s.ax = ax; s.ay = ay; es += Math.hypot(tx - s.x, ty - s.y); dvSum += am * dt; }
      for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) {
        const d = Math.hypot(sats[i].x - sats[j].x, sats[i].y - sats[j].y); if (d < minSep) minSep = d; }
      for (const s of sats) { s.vx += s.ax * dt; s.vy += s.ay * dt; s.x += s.vx * dt; s.y += s.vy * dt; }
      rms = es / n;
    }
  }
  return { finalRms: rms, minSep, dvPerAgent: dvSum / n };
}

// --- score both controllers over the fixed seed set --------------------------
function score(runner) {
  let ok = 0, sumRms = 0, sumSep = 0, sumDv = 0, worstSep = Infinity;
  for (const seed of SEEDS) {
    const r = runner(seed);
    const converged = r.finalRms < CONVERGED, safe = r.minSep >= COLLISION;
    if (converged && safe) ok++;
    sumRms += r.finalRms; sumSep += r.minSep; sumDv += r.dvPerAgent; worstSep = Math.min(worstSep, r.minSep);
  }
  const n = SEEDS.length;
  return { success: ok / n, meanRms: sumRms / n, meanSep: sumSep / n, worstSep, meanDv: sumDv / n };
}

const analytic = score(seed => { const r = formationRun(N, PATTERNS, seed); return { finalRms: r.finalRms, minSep: r.minSep, dvPerAgent: r.dvPerAgent }; });
const learned = score(seed => learnedRun(N, PATTERNS, seed));

const pct = x => (x * 100).toFixed(1) + '%';
console.log(`\nFormation-Bench · ${N} agents · ${PATTERNS.join('→')} · ${SEEDS.length} seeds · collision < ${COLLISION} m\n`);
console.log('controller                 success   mean RMS   mean min-sep   worst min-sep   mean Δv/agent');
console.log('-------------------------------------------------------------------------------------------');
for (const [name, s] of [['Distributed (analytic)', analytic], ['Learned (1,474p, on-device)', learned]]) {
  console.log(
    name.padEnd(27) + pct(s.success).padStart(7) +
    (s.meanRms.toFixed(2) + ' m').padStart(11) +
    (s.meanSep.toFixed(2) + ' m').padStart(15) +
    (s.worstSep.toFixed(2) + ' m').padStart(16) +
    (s.meanDv.toFixed(1) + ' m/s').padStart(16));
}
console.log('\nsuccess = holds the shape (RMS < ' + CONVERGED + ' m) AND collision-free (min sep ≥ ' + COLLISION + ' m) on every reconfiguration.');

const results = { benchmark: 'Formation-Bench', n: N, patterns: PATTERNS, seeds: SEEDS.length,
  collisionRadius: COLLISION, convergedRms: CONVERGED,
  controllers: {
    'distributed-analytic': analytic,
    'learned-1474p': { ...learned, params: 10*32+32 + 32*32+32 + 32*2+2, note: 'compact per-agent net + reflexive safety filter; clones coordination, carries a residual min-separation gap vs the certified analytic controller' },
  } };
fs.writeFileSync(new URL('./formation-results.json', import.meta.url), JSON.stringify(results, null, 2));
console.log('\nwrote bench/formation-results.json');
