// Sanity checks against independently re-derived values. `node --test core/`
import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  tetherMassRatio, sledTrackLength, sledPeakPower, sledEnergyPerKg,
  skyhook, escapeVelocity, orbitalVelocity, compose,
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
