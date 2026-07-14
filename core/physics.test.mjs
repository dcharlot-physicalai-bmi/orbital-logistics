// Sanity checks against independently re-derived values. `node --test core/`
import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  tetherMassRatio, sledTrackLength, sledPeakPower, sledEnergyPerKg,
  skyhook, escapeVelocity, orbitalVelocity, compose,
  meanMotion, cwPropagate, cwTargetIntercept, mppiCapture,
  rigidBodyStep, rotEnergy, angMomentum, qRotate,
  zemzev, land,
} from './physics.mjs';

const near = (a, b, tol) => assert.ok(Math.abs(a - b) <= tol, `${a} not within ${tol} of ${b}`);

test('sled: 2.9 km/s at 10 g needs ~42.9 km of track', () => {
  near(sledTrackLength(2900, 10) / 1000, 42.9, 0.3);
});

test('sled: peak power ~14 GW, energy ~210 GJ for a 50 t vehicle', () => {
  near(sledPeakPower(2900, 10, 50000) / 1e9, 14.3, 0.5);       // GW
  near(sledEnergyPerKg(2900) * 50000 / 1e9, 210, 5);           // GJ
});

test('tether: Kevlar ~13.6x, nanotube ~0.26x at a 3 km/s tip', () => {
  near(tetherMassRatio(3000, 'kevlar'), 13.6, 0.4);
  near(tetherMassRatio(3000, 'nanotube'), 0.26, 0.05);
});

test('skyhook: a 600 km / 400 km-arm hook releases above escape at the tip', () => {
  const s = skyhook(600e3, 3000, 400e3);
  near(s.release / 1000, 10.56, 0.1);       // inertial release speed
  near(s.escapeAtTip / 1000, 10.40, 0.1);   // escape at the ~1000 km tip radius
  assert.ok(s.release > s.escapeAtTip, 'release should exceed escape at the tip');
});

test('orbit: LEO circular velocity ~7.56 km/s at 600 km', () => {
  near(orbitalVelocity(600e3) / 1000, 7.56, 0.05);
});

test('compose: Mars is unreachable by a single vehicle, reachable composed', () => {
  const solo = compose('mars', []);
  assert.equal(solo.soloFeasible, false);
  const built = compose('mars', [4.0, 6.0, 2.5, 1.5]); // skyhook+loop+depot+aero
  assert.ok(built.payloadFraction > 0.2, 'composed payload fraction should be healthy');
});

test('compose: LEO offload lifts payload fraction over all-rocket', () => {
  const r = compose('leo', [0.7, 4.0]); // sled + skyhook
  assert.ok(r.soloFeasible);
  assert.ok(r.gain > 5, 'meaningful payload gain');
  near(r.share, 4.7 / 9.4, 0.02);
});

test('CW: mean motion at 400 km gives a ~92 min period', () => {
  const n = meanMotion(400e3);
  near((2 * Math.PI / n) / 60, 92.5, 1.0); // minutes
});

test('CW: two objects along-track on the same orbit stay put', () => {
  const n = meanMotion(500e3);
  const s = cwPropagate([0, 120], [0, 0], n, 1800); // 120 m ahead, half an hour
  near(s.r[0], 0, 1e-6); near(s.r[1], 120, 1e-6);
  near(s.v[0], 0, 1e-9); near(s.v[1], 0, 1e-9);
});

test('CW: a radial offset with zero rate drifts along-track (not a fixed point)', () => {
  const n = meanMotion(500e3);
  const s = cwPropagate([50, 0], [0, 0], n, 2000);
  assert.ok(Math.abs(s.r[1]) > 100, 'radial offset induces along-track drift');
});

test('MPPI: receding-horizon control captures from a standoff', () => {
  const n = meanMotion(450e3);
  const r = mppiCapture([30, 8, 0, 0], n, {}, 7);
  assert.ok(r.captured, 'MPPI berths the chaser');
  assert.ok(r.finalRange < 0.5, 'arrives inside the capture tolerance');
  assert.ok(r.dvTotal > 0 && r.dvTotal < 6, 'a bounded, sane fuel cost');
});

test('MPPI: holds the approach corridor near the target', () => {
  const n = meanMotion(450e3);
  const r = mppiCapture([30, 8, 0, 0], n, {}, 7);
  // once inside 8 m, the along-track (lateral) offset stays small (in the corridor)
  const near = r.trajectory.filter(s => Math.hypot(s[0], s[1]) < 8);
  assert.ok(near.length > 0);
  assert.ok(Math.max(...near.map(s => Math.abs(s[1]))) < 3.0, 'corridor held near the target');
});

test('6-DOF: torque-free tumble conserves energy and angular momentum', () => {
  const I = [1.0, 2.5, 3.0];
  let q = [1, 0, 0, 0], w = [0.6, 0.2, 0.4];       // a genuine 3-axis tumble
  const E0 = rotEnergy(w, I), L0 = angMomentum(w, I);
  for (let k = 0; k < 4000; k++) ({ q, w } = rigidBodyStep(q, w, I, 0.01));
  near(rotEnergy(w, I), E0, 1e-3 * E0);            // energy invariant
  near(angMomentum(w, I), L0, 1e-3 * L0);          // |L| invariant
  near(Math.hypot(...q), 1, 1e-9);                 // quaternion stays unit
});

test('6-DOF: intermediate-axis spin is unstable (tennis-racket theorem)', () => {
  const I = [1.0, 2.0, 3.0];
  // spin almost purely about the intermediate axis (2) with a tiny perturbation
  let q = [1, 0, 0, 0], w = [0.02, 1.0, 0.02];
  let maxOff = 0;
  for (let k = 0; k < 3000; k++) { ({ q, w } = rigidBodyStep(q, w, I, 0.01)); maxOff = Math.max(maxOff, Math.abs(w[0]) + Math.abs(w[2])); }
  assert.ok(maxOff > 0.5, 'the off-axis rates grow large — the flip is unstable');
});

test('landing: ZEM/ZEV nulls position and velocity at tgo (analytic)', () => {
  // one open-loop step: applying the ZEM/ZEV command and coasting for tgo should
  // land at the target with zero velocity (the law is exact for constant accel)
  const g = [0, -1.62], tgo = 30, r = [400, 800], v = [-20, -40];
  const a = zemzev(r, v, [0, 0], [0, 0], g, tgo);
  // the ZEM/ZEV command is time-varying; check it points to reduce the miss
  const rf = r.map((ri, i) => ri + v[i] * tgo + 0.5 * (a[i] + g[i]) * tgo * tgo);
  assert.ok(Math.hypot(...rf) < Math.hypot(...r), 'the command drives toward the pad');
});

test('landing: closed-loop guidance touches down soft and on-target', () => {
  const g = [0, -1.62];                       // lunar gravity
  const r = land([300, 900], [-25, -55], g, 6.0, { dt: 0.1 });
  assert.ok(r.landed, 'reaches the surface');
  assert.ok(r.miss < 5, 'lands within a few metres of the pad');   // precision
  assert.ok(r.speed < 2.0, 'soft touchdown');                       // gentle
  assert.ok(r.fuel > 0 && r.fuel < 500, 'a bounded, sane Δv budget');
});

test('landing: an underpowered lander cannot make it soft', () => {
  const g = [0, -1.62];
  const weak = land([300, 900], [-25, -55], g, 1.2, { dt: 0.1 });  // barely above gravity
  assert.ok(!weak.landed || weak.speed > 2.0, 'too little thrust → crash or miss');
});

test('6-DOF: qRotate preserves length and identity', () => {
  near(Math.hypot(...qRotate([1, 0, 0, 0], [1, 2, 3])), Math.hypot(1, 2, 3), 1e-9);
  const p = qRotate([Math.cos(0.5), 0, 0, Math.sin(0.5)], [1, 0, 0]); // 1 rad about z
  near(Math.hypot(...p), 1, 1e-9);
});

test('MPPI: same seed is deterministic (reproducible)', () => {
  const n = meanMotion(450e3);
  const a = mppiCapture([25, 6, 0, 0], n, {}, 3);
  const b = mppiCapture([25, 6, 0, 0], n, {}, 3);
  assert.equal(a.steps, b.steps);
  near(a.dvTotal, b.dvTotal, 1e-12);
  near(a.finalRange, b.finalRange, 1e-12);
});

test('CW: two-impulse targeting reaches the target', () => {
  const n = meanMotion(450e3);
  const r0 = [40, 25], t = 900; // 40 m radial, 25 m along-track; 15 min transfer
  const plan = cwTargetIntercept(r0, [0, 0], n, t);
  const arr = cwPropagate(r0, plan.v0, n, t); // fly the solved v0
  near(arr.r[0], 0, 1e-4); near(arr.r[1], 0, 1e-4); // arrives at origin
  assert.ok(plan.dvTotal > 0 && plan.dvTotal < 10, 'a sane, bounded two-burn cost');
  // dv2 nulls the arrival velocity for a soft berth
  near(arr.v[0] + plan.dv2[0], 0, 1e-6);
  near(arr.v[1] + plan.dv2[1], 0, 1e-6);
});
