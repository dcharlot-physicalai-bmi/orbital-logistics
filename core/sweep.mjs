#!/usr/bin/env node
// orbital-logistics · headless parameter sweeps over the physics core.
//   node core/sweep.mjs tether      tether mass ratio vs tip speed, per material
//   node core/sweep.mjs sled        track length / power vs exit speed
//   node core/sweep.mjs skyhook     catch/release velocities vs altitude
//   node core/sweep.mjs compose     best-known architectures per destination
import {
  TETHER_MATERIALS, tetherMassRatio, sledTrackLength, sledPeakPower,
  skyhook, escapeVelocity, DESTINATIONS, compose,
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
} else {
  console.log('usage: node core/sweep.mjs [tether|sled|skyhook|compose]');
}
