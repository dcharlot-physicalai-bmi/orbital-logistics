// gen_land_data.mjs — distill the clock-free optimal descent guidance into demonstrations.
//
// The expert is core.landGuidance: at every state it solves for the energy-optimal
// time-to-go and returns the ZEM/ZEV thrust. That makes it a genuine STATE-FEEDBACK
// law — the same state always yields the same thrust — so cloning it is well-posed.
// (Cloning the schedule-flown land() instead is NOT: two descents through the same
// state with different time-to-go want different thrust, and the net averages them
// into a lander that arrives on-target but too fast.)
//
// The expert still pays a scalar time-to-go minimisation every step. The net replaces
// it with one forward pass. State = [dx, dy, vx, vy, g, aMax] — position relative to the
// pad, velocity, the body's gravity, the engine limit — so one net spans Moon/Mars/Earth.
// Output: policy/land_data.json
import { writeFileSync } from 'node:fs';
import { landGuidance } from '../core/physics.mjs';

function mulberry32(a){ return function(){ a|=0; a=a+0x6D2B79F5|0; let t=Math.imul(a^a>>>15,1|a);
  t=t+Math.imul(t^t>>>7,61|t)^t; return ((t^t>>>14)>>>0)/4294967296; }; }

const rows = [];                                 // each: [dx, dy, vx, vy, g, aMax, ax, ay]
const dt = 0.1;
// Roll the clock-free law out, recording (state -> thrust). Returns whether it landed
// soft; only successful descents are kept, so the net learns from good demonstrations.
function record(r0, v0, grav, aMax) {
  const g = [0, -grav];
  let r = r0.slice(), v = v0.slice(); const buf = [];
  for (let k = 0; k < 4000; k++) {
    const a = landGuidance(r, v, g, aMax);       // pure state feedback, no clock
    buf.push([r[0], r[1], v[0], v[1], grav, aMax, a[0], a[1]]);
    v = [v[0] + (a[0] + g[0]) * dt, v[1] + (a[1] + g[1]) * dt];
    r = [r[0] + v[0] * dt, r[1] + v[1] * dt];
    if (r[1] <= 0) {
      const soft = Math.abs(r[0]) < 8 && Math.hypot(...v) < 2.0;
      if (soft) rows.push(...buf);
      return soft;
    }
  }
  return false;
}

const BODIES = [1.62, 3.71, 9.81];               // Moon, Mars, Earth
let landed = 0, total = 0;
for (let seed = 1; seed <= 900; seed++) {
  const rng = mulberry32(seed * 41 + 3);
  const grav = BODIES[seed % BODIES.length];
  const twr = 1.8 + rng() * 1.9, aMax = twr * grav;          // engine 1.8–3.7× g
  const alt = 200 + rng() * 1000;                             // 200–1200 m up
  const lat = (rng() * 2 - 1) * 400;                          // ±400 m off the pad
  const vx = (rng() * 2 - 1) * 30, vy = -(10 + rng() * 60);   // coming down
  total++;
  if (record([lat, alt], [vx, vy], grav, aMax)) landed++;
}
writeFileSync(new URL('./land_data.json', import.meta.url),
  JSON.stringify({ in: 6, out: 2, n: rows.length, rows }));
console.log(`landing demonstrations: ${rows.length} samples from ${landed}/${total} soft clock-free descents`);
