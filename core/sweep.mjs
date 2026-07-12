#!/usr/bin/env node
// orbital-logistics · headless parameter sweeps over the physics core.
//   node core/sweep.mjs tether      tether mass ratio vs tip speed, per material
//   node core/sweep.mjs sled        track length / power vs exit speed
//   node core/sweep.mjs skyhook     catch/release velocities vs altitude
//   node core/sweep.mjs compose     best-known architectures per destination
//   node core/sweep.mjs rpo         two-impulse CW rendezvous cost vs transfer time
//   node core/sweep.mjs mppi        MPPI receding-horizon capture from several standoffs
import {
  TETHER_MATERIALS, tetherMassRatio, sledTrackLength, sledPeakPower,
  skyhook, escapeVelocity, DESTINATIONS, compose,
  meanMotion, cwTargetIntercept, mppiCapture,
} from './physics.mjs';

const mode = process.argv[2] || 'help';
const pad = (s, n) => String(s).padStart(n);
const row = (...c) => console.log(c.join('  '));

if (mode === 'tether') {
  console.log('# tether mass / payload  (constant-stress taper)');
  row(pad('Vtip km/s', 9), ...Object.values(TETHER_MATERIALS).map(m => pad(m.name, 16)));
  for (let v = 1000; v <= 5000; v += 1000) {
    row(pad((v / 1000).toFixed(1), 9),
      ...Object.keys(TETHER_MATERIALS).map(k => {
        const r = tetherMassRatio(v, k);
        return pad(r > 1e4 ? r.toExponential(1) : r.toFixed(2) + 'x', 16);
      }));
  }
} else if (mode === 'sled') {
  console.log('# maglev sled (50 t vehicle)');
  row(pad('exit km/s', 9), pad('10 g track', 12), pad('20 g track', 12), pad('10 g peak', 12));
  for (let v = 500; v <= 3000; v += 500) {
    row(pad((v / 1000).toFixed(1), 9),
      pad((sledTrackLength(v, 10) / 1000).toFixed(1) + ' km', 12),
      pad((sledTrackLength(v, 20) / 1000).toFixed(1) + ' km', 12),
      pad((sledPeakPower(v, 10, 50000) / 1e9).toFixed(1) + ' GW', 12));
  }
} else if (mode === 'skyhook') {
  console.log('# rotovator, 3 km/s tip — catch/release vs altitude');
  row(pad('alt km', 7), pad('orbital', 9), pad('catch', 9), pad('release', 9), pad('escape', 9), 'note');
  for (const alt of [400, 600, 1000, 1500]) {
    const s = skyhook(alt * 1e3, 3000), esc = skyhook(alt * 1e3, 3000).escapeAtTip / 1000;
    row(pad(alt, 7), pad((s.orbital / 1e3).toFixed(2), 9), pad((s.catch / 1e3).toFixed(2), 9),
      pad((s.release / 1e3).toFixed(2), 9), pad(esc.toFixed(2), 9),
      s.release / 1e3 > esc ? 'release > escape' : '');
  }
} else if (mode === 'compose') {
  console.log('# a strong known architecture per destination (sled+skyhook+depot+aero as needed)');
  const plans = { leo: [0.7, 4.0], geo: [0.7, 4.0, 2.5], moon: [0.7, 4.0, 6.0], mars: [0.7, 4.0, 6.0, 2.5, 1.5] };
  row(pad('dest', 14), pad('budget', 8), pad('offload', 8), pad('onboard', 8), pad('payload', 8), 'vs single vehicle');
  for (const k of Object.keys(DESTINATIONS)) {
    const r = compose(k, plans[k]);
    row(pad(DESTINATIONS[k].name, 14), pad(r.total.toFixed(1), 8), pad(r.offloaded.toFixed(1), 8),
      pad(r.onboard.toFixed(1), 8), pad((r.payloadFraction * 100).toFixed(1) + '%', 8),
      r.soloFeasible ? '×' + r.gain.toFixed(1) : 'unreachable alone');
  }
} else if (mode === 'rpo') {
  console.log('# two-impulse Clohessy-Wiltshire rendezvous from 100 m (radial+along-track) at 450 km');
  const n = meanMotion(450e3), r0 = [70, 70];
  row(pad('transfer min', 12), pad('dv1 m/s', 9), pad('dv2 m/s', 9), pad('total m/s', 10));
  for (const min of [5, 10, 20, 40, 60]) {
    const p = cwTargetIntercept(r0, [0, 0], n, min * 60);
    const m = (u) => Math.hypot(u[0], u[1]).toFixed(3);
    row(pad(min, 12), pad(m(p.dv1), 9), pad(m(p.dv2), 9), pad(p.dvTotal.toFixed(3), 10));
  }
  console.log('# slower transfers cost less dv (a gentler catch), the tradeoff the certified corridor flies.');
} else if (mode === 'mppi') {
  console.log('# MPPI receding-horizon capture at 450 km, various standoffs (radial, along-track)');
  const n = meanMotion(450e3);
  row(pad('standoff m', 22), pad('captured', 9), pad('steps', 6), pad('final m', 8), pad('dv m/s', 8));
  for (const s0 of [[20, 4, 0, 0], [30, 8, 0, 0], [45, 12, 0, 0], [60, 20, 0, 0]]) {
    const r = mppiCapture(s0, n, {}, 7);
    row(pad(`(${s0[0]}, ${s0[1]})`, 22), pad(r.captured ? 'yes' : 'NO', 9),
      pad(r.steps, 6), pad(r.finalRange.toFixed(2), 8), pad(r.dvTotal.toFixed(2), 8));
  }
  console.log('# a real optimal controller sampled and rolled out on the CW dynamics, holding the corridor.');
} else {
  console.log('usage: node core/sweep.mjs [tether|sled|skyhook|compose|rpo|mppi]');
}
