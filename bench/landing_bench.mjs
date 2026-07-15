// landing_bench.mjs — an open, deterministic benchmark for autonomous powered descent.
// 120 fixed starts (40 per body: Moon / Mars / Earth), varying altitude, downrange
// offset, entry velocity and engine thrust-to-weight. Scored on the three things that
// matter: did it land soft (< 2 m/s), on the pad (< 8 m), and at what fuel cost.
// Bit-identical every run.
//
// Four guidance schemes, and the comparison is the finding:
//   Scheduled ZEM/ZEV — the classic: search a flight time, then fly the clock down.
//   Clock-free ZEM/ZEV — solve the optimal time-to-go from the state at every step;
//                        no clock, no plan, and far more precise.
//   Learned (thrust)   — clone the clock-free law's THRUST directly. Fails: the gains
//                        go as 6/tgo², so the target is stiff exactly at touchdown.
//   Learned (tgo)      — clone only the time-to-go and keep the closed form. Matches
//                        the expert at 1,281 parameters, as one forward pass.
import fs from 'node:fs';
import { land, landFeedback, landGuidance, zemzev, optimalTgo } from '../core/physics.mjs';

const dt = 0.1, BODIES = [[1.62, 'Moon'], [3.71, 'Mars'], [9.81, 'Earth']];
function mulberry32(a){ return function(){ a|=0; a=a+0x6D2B79F5|0; let t=Math.imul(a^a>>>15,1|a);
  t=t+Math.imul(t^t>>>7,61|t)^t; return ((t^t>>>14)>>>0)/4294967296; }; }
function makeIC(seed, grav) {
  const rng = mulberry32(seed * 41 + 3);
  const twr = 1.8 + rng() * 1.9, aMax = twr * grav;
  const alt = 200 + rng() * 1000, lat = (rng() * 2 - 1) * 400;
  const vx = (rng() * 2 - 1) * 30, vy = -(10 + rng() * 60);
  return { r0: [lat, alt], v0: [vx, vy], grav, aMax };
}
const SEEDS = Array.from({ length: 40 }, (_, i) => 5000 + i);   // held out of every training set

// --- the learned nets -------------------------------------------------------
const relu = z => z.map(v => Math.max(0, v));
const matvec = (W, x) => W[0].map((_, j) => x.reduce((s, xi, i) => s + xi * W[i][j], 0));
function loadNet(file) {
  const u = new URL(`../policy/${file}`, import.meta.url);
  if (!fs.existsSync(u)) return null;
  const P = JSON.parse(fs.readFileSync(u));
  const fwd = sx => { const x = sx.map((v, j) => (v - P.xm[j]) / P.xsd[j]);
    const a1 = relu(matvec(P.W1, x).map((v, j) => v + P.b1[j]));
    const a2 = relu(matvec(P.W2, a1).map((v, j) => v + P.b2[j]));
    return matvec(P.W3, a2).map((v, j) => v + P.b3[j]).map((v, j) => v * P.ysd[j] + P.ym[j]); };
  return { P, fwd };
}
const tgoNet = loadNet('land_policy.json');                       // mode: tgo (shipped)
const thrustNet = loadNet('land_policy_thrust_dagger.json');      // optional baseline

// --- the four controllers, each: IC -> { landed, miss, speed, fuel } ---------
function flyClosedLoop(c, action) {
  let r = c.r0.slice(), v = c.v0.slice(), fuel = 0;
  for (let k = 0; k < 4000; k++) {
    let a = action(r, v, c);
    const am = Math.hypot(...a); if (am > c.aMax) a = a.map(x => x * c.aMax / am);
    v = [v[0] + a[0] * dt, v[1] + (a[1] - c.grav) * dt];
    r = [r[0] + v[0] * dt, r[1] + v[1] * dt];
    fuel += Math.hypot(...a) * dt;
    if (r[1] <= 0) return { landed: Math.abs(r[0]) < 8 && Math.hypot(...v) < 2.0, miss: Math.abs(r[0]), speed: Math.hypot(...v), fuel };
  }
  return { landed: false, miss: Math.abs(r[0]), speed: Math.hypot(...v), fuel };
}
const CONTROLLERS = {
  'Scheduled ZEM/ZEV': c => { const r = land(c.r0, c.v0, [0, -c.grav], c.aMax, { dt });
    return { landed: r.landed, miss: r.miss, speed: r.speed, fuel: r.fuel }; },
  'Clock-free ZEM/ZEV': c => { const r = landFeedback(c.r0, c.v0, [0, -c.grav], c.aMax, { dt });
    return { landed: r.landed, miss: r.miss, speed: r.speed, fuel: r.fuel }; },
  'Learned · thrust (1,346p)': !thrustNet ? null :
    c => flyClosedLoop(c, (r, v) => thrustNet.fwd([r[0], r[1], v[0], v[1], c.grav, c.aMax])),
  'Learned · tgo (1,281p)': !tgoNet ? null :
    c => flyClosedLoop(c, (r, v) => {
      const tgo = Math.max(0.5, tgoNet.fwd([r[0], r[1], v[0], v[1], c.grav])[0]);
      return zemzev(r, v, [0, 0], [0, 0], [0, -c.grav], tgo);
    }),
};

const results = {};
for (const [name, run] of Object.entries(CONTROLLERS)) {
  if (!run) { console.log(`(skipping ${name}: policy file absent — run its trainer first)`); continue; }
  let ok = 0, tot = 0, ms = 0, sp = 0, fu = 0; const perBody = {};
  for (const [grav, body] of BODIES) {
    let bok = 0;
    for (const seed of SEEDS) { const r = run(makeIC(seed, grav)); tot++;
      if (r.landed) { ok++; bok++; ms += r.miss; sp += r.speed; fu += r.fuel; } }
    perBody[body] = `${bok}/${SEEDS.length}`;
  }
  results[name] = { success: ok / tot, landed: ok, of: tot, perBody,
    meanMiss: ok ? ms / ok : null, meanTouchdown: ok ? sp / ok : null, meanFuel: ok ? fu / ok : null };
}

console.log(`\nLanding-Bench · ${SEEDS.length * BODIES.length} fixed starts · Moon/Mars/Earth · soft = <2 m/s and <8 m\n`);
console.log('guidance                     success   Moon   Mars   Earth    mean miss   touchdown   mean Δv');
console.log('---------------------------------------------------------------------------------------------');
for (const [name, r] of Object.entries(results)) {
  console.log(name.padEnd(28) + ((r.success * 100).toFixed(1) + '%').padStart(7) +
    r.perBody.Moon.padStart(7) + r.perBody.Mars.padStart(7) + r.perBody.Earth.padStart(8) +
    (r.meanMiss.toFixed(2) + ' m').padStart(12) + (r.meanTouchdown.toFixed(2) + ' m/s').padStart(12) +
    (r.meanFuel.toFixed(0) + ' m/s').padStart(10));
}
console.log('\nmean miss / touchdown / Δv are over the landings that succeeded.');

fs.writeFileSync(new URL('./landing-results.json', import.meta.url),
  JSON.stringify({ benchmark: 'Landing-Bench', starts: SEEDS.length * BODIES.length,
    bodies: BODIES.map(b => b[1]), soft: { speed: 2.0, miss: 8 }, controllers: results }, null, 2));
console.log('wrote bench/landing-results.json');
