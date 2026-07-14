// gen_formation_data.mjs — distill the distributed formation controller.
// Roll out the analytic controller (core/physics.mjs) across many seeds and
// patterns and record, for EVERY agent at EVERY step, its LOCAL observation and
// the action the controller took. The observation is decentralized by
// construction — an agent sees only its own slot error, its velocity relative to
// the slot, and its three nearest neighbours — so a policy cloned from it inherits
// the same "no one in charge" property. Output: policy/formation_data.json
import { writeFileSync } from 'node:fs';
import { FORMATION_DEFAULTS as D, formationSlots, assignSlots, scatterAgents } from '../core/physics.mjs';

function mulberry32(a){ return function(){ a|=0; a=a+0x6D2B79F5|0; let t=Math.imul(a^a>>>15,1|a);
  t=t+Math.imul(t^t>>>7,61|t)^t; return ((t^t>>>14)>>>0)/4294967296; }; }

const K = 3;                                  // neighbours in the observation
const rows = [];                              // each: [ ...obs(10), ax, ay ]

// One instrumented step: compute each agent's local obs + the controller's action.
function stepAndRecord(sats, slots, assign, rot, dt, record) {
  const c = Math.cos(rot), si = Math.sin(rot);
  for (let i = 0; i < sats.length; i++) {
    const s = sats[i], sl = slots[assign[i]];
    const tx = sl[0]*c - sl[1]*si, ty = sl[0]*si + sl[1]*c;
    const svx = -D.rot*ty, svy = D.rot*tx;
    // the same local control law as core.formationStep:
    let ax = D.kp*(tx-s.x) - D.kd*(s.vx-svx), ay = D.kp*(ty-s.y) - D.kd*(s.vy-svy);
    const nb = [];
    for (let j = 0; j < sats.length; j++) { if (j===i) continue;
      const dx = s.x-sats[j].x, dy = s.y-sats[j].y, d = Math.hypot(dx,dy);
      nb.push({ dx, dy, d });
      if (d < D.senseR) { const w = (D.senseR-d)/D.senseR; ax += dx/(d||1e-9)*w*w*D.repel; ay += dy/(d||1e-9)*w*w*D.repel; }
    }
    const am = Math.hypot(ax,ay); if (am > D.umax) { ax *= D.umax/am; ay *= D.umax/am; }
    s.ax = ax; s.ay = ay;
    if (record) {
      nb.sort((p,q) => p.d - q.d);
      const near = [];
      for (let k = 0; k < K; k++) { const n = nb[k] || { dx: D.senseR*3, dy: 0 }; near.push(n.dx, n.dy); }
      // obs (agent frame is world-aligned; the rotation enters through the slot error):
      rows.push([ tx-s.x, ty-s.y, s.vx-svx, s.vy-svy, ...near, ax, ay ]);
    }
  }
  for (const s of sats) { s.vx += s.ax*dt; s.vy += s.ay*dt; s.x += s.vx*dt; s.y += s.vy*dt; }
}

const PATTERNS = ['ring', 'grid', 'wedge', 'line'];
const dt = D.dt, holdSteps = Math.round(D.hold/dt);
let steps = 0;
for (let seed = 1; seed <= 60; seed++) {
  const n = 8 + (seed % 7);                    // vary the crowd size 8..14
  const rng = mulberry32(seed*97 + 5);
  const sats = scatterAgents(n, D.R, rng, 18);
  let rot = 0;
  // a shuffled pattern order per seed, so transfers of every kind are seen
  const order = [...PATTERNS].sort(() => rng() - 0.5);
  for (const p of order) {
    const slots = formationSlots(p, n, D.R), assign = assignSlots(sats, slots, rot);
    for (let k = 0; k < holdSteps; k++) { rot += D.rot*dt;
      // subsample: transfers (early in a hold) are richer than the held tail
      stepAndRecord(sats, slots, assign, rot, dt, k % 2 === 0); steps++;
    }
  }
}
writeFileSync(new URL('./formation_data.json', import.meta.url),
  JSON.stringify({ K, in: 10, out: 2, n: rows.length, rows }));
console.log(`formation demonstrations: ${rows.length} samples from ${steps} controller steps`);
