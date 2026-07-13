// Generate a behavior-cloning dataset from the MPPI expert: for many capture
// rollouts, record (relative state -> optimal control) pairs in the corridor
// frame (x = along the approach axis, y = lateral). A small policy is then
// distilled from this to fly the approach on-device.
import { meanMotion, cwStepEuler, mppiPlan } from '../core/physics.mjs';
import fs from 'node:fs';

const n = meanMotion(450e3);
const DT = 1.0;            // matches the core MPPI timestep
const RUNS = 240;
const data = [];          // [x, y, vx, vy, ax, ay]
let seed = 1, captured = 0;

for (let run = 0; run < RUNS; run++) {
  let s = [14 + Math.random() * 34, (Math.random() * 2 - 1) * 16,
           (Math.random() * 2 - 1) * 0.3, (Math.random() * 2 - 1) * 0.3];
  let nominal = Array.from({ length: 28 }, () => [0, 0]);
  for (let step = 0; step < 260; step++) {
    nominal = mppiPlan(s, nominal, n, {}, seed++);
    const u = nominal[0];
    data.push([s[0], s[1], s[2], s[3], u[0], u[1]]);
    s = cwStepEuler(s, u, n, DT);
    nominal = [...nominal.slice(1), [0, 0]];
    const range = Math.hypot(s[0], s[1]), speed = Math.hypot(s[2], s[3]);
    if (range < 0.5 && speed < 0.05 && Math.abs(s[1]) < 0.4) { captured++; break; }
  }
}

fs.writeFileSync(new URL('./bc_data.json', import.meta.url),
  JSON.stringify({ cols: ['x', 'y', 'vx', 'vy', 'ax', 'ay'], rows: data, dt: DT }));
console.log(`wrote bc_data.json — ${data.length} samples from ${RUNS} rollouts (${captured} captured)`);
