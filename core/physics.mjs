// orbital-logistics · headless physics core
// Institute for Physical AI @ BMI · Technical Report TR-2026-17
//
// The load-bearing equations behind the instruments, extracted so they run
// headless for parameter sweeps. Every function is the same math the in-browser
// instrument draws; nothing here is fitted or fabricated. SI units unless noted.

export const G0 = 9.80665;       // m/s^2, standard gravity
export const MU_EARTH = 3.986e14; // m^3/s^2, Earth gravitational parameter
export const R_EARTH = 6.371e6;   // m, mean Earth radius

// Abramowitz & Stegun 7.1.26 error-function approximation (|err| < 1.5e-7).
export function erf(x) {
  const s = x < 0 ? -1 : 1; x = Math.abs(x);
  const t = 1 / (1 + 0.3275911 * x);
  const y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t
    - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x);
  return s * y;
}

// --- Momentum-exchange tether (skyhook / rotovator) ------------------------
// Tether characteristic velocity Vc = sqrt(2 * sigma / rho) sets the scale.
export const TETHER_MATERIALS = {
  kevlar:   { name: 'Kevlar-49', vc: 2236 },
  spectra:  { name: 'Spectra',   vc: 2648 },
  zylon:    { name: 'Zylon',     vc: 2727 },
  nanotube: { name: 'Carbon nanotube', vc: 8637 },
};

// Constant-stress taper: ratio of tether mass to payload mass to sustain a tip
// speed Vtip in a material of characteristic velocity Vc.
//   m_tether / m_payload = sqrt(pi) * x * e^{x^2} * erf(x),  x = Vtip / Vc
export function tetherMassRatio(vtip, materialKey) {
  const vc = TETHER_MATERIALS[materialKey].vc;
  const x = vtip / vc;
  return Math.sqrt(Math.PI) * x * Math.exp(x * x) * erf(x);
}

// Circular orbital velocity at a given altitude (m). vis-viva, circular case.
export function orbitalVelocity(altitude_m) {
  return Math.sqrt(MU_EARTH / (R_EARTH + altitude_m));
}

// A rotovator catches at Vcatch = Vorbit - Vtip and releases at Vorbit + Vtip
// (tip speeds relative to the station; first-order, ignoring L*Omega corrections).
// Release happens at the tip, one arm length above the station, so the payload is
// compared against escape velocity AT THE TIP RADIUS, not at the station. All m/s.
export function skyhook(altitude_m, vtip, armLength_m = 400e3) {
  const vo = orbitalVelocity(altitude_m);
  const tipAlt = altitude_m + armLength_m;
  return {
    orbital: vo, catch: vo - vtip, release: vo + vtip,
    tipAltitude: tipAlt, escapeAtTip: escapeVelocity(tipAlt),
  };
}

// Local escape velocity at radius R_EARTH + altitude.
export function escapeVelocity(altitude_m) {
  return Math.sqrt(2 * MU_EARTH / (R_EARTH + altitude_m));
}

// --- Electromagnetic launch (maglev sled / mass driver) --------------------
// Track length to reach an exit speed at a fixed g-load: L = v^2 / (2 a).
export function sledTrackLength(exitSpeed, gLoad) {
  const a = gLoad * G0;
  return (exitSpeed * exitSpeed) / (2 * a); // m
}
// Kinetic launch energy per unit vehicle mass, J/kg.
export function sledEnergyPerKg(exitSpeed) { return 0.5 * exitSpeed * exitSpeed; }
// Peak electrical power at exit for a vehicle of mass m (kg): P = m a v.
export function sledPeakPower(exitSpeed, gLoad, mass_kg) {
  return mass_kg * (gLoad * G0) * exitSpeed; // W
}

// --- Delta-v budget composer ------------------------------------------------
// Representative velocity budgets from the ground, km/s (incl. losses/arrival).
export const DESTINATIONS = {
  leo:  { name: 'Low Earth orbit', dv: 9.4 },
  geo:  { name: 'Geostationary',   dv: 13.3 },
  moon: { name: 'Lunar surface',   dv: 15.2 },
  mars: { name: 'Mars surface',    dv: 18.0 },
};
export const VE = 3.55; // km/s, effective exhaust velocity of the onboard stack

// Compose an architecture: a list of offloaded segment delta-vs (km/s) against a
// destination. Returns the honest budget, incl. payload mass fraction. The
// mass-ratio math is the exact rocket equation; a single-vehicle baseline that
// cannot physically reach the destination is reported as unreachable, not as an
// inflated multiplier.
export function compose(destKey, segments, opts = {}) {
  const eps = opts.eps ?? 0.06; // inert (structure) mass fraction
  const T = DESTINATIONS[destKey].dv;
  const off = Math.min(segments.reduce((a, b) => a + b, 0), T - 0.1);
  const onboard = Math.max(0.1, T - off);
  const MR = Math.exp(onboard / VE), MR0 = Math.exp(T / VE);
  const lam = 1 / MR - eps;    // payload fraction, composed
  const lam0 = 1 / MR0 - eps;  // payload fraction, single vehicle
  const soloFeasible = lam0 > 0.002;
  return {
    total: T, offloaded: off, onboard, massRatio: MR,
    payloadFraction: lam, soloFeasible,
    gain: soloFeasible ? lam / lam0 : null,
    share: off / T,
  };
}

// --- Capture GNC: per-axis constant-velocity Kalman filter -----------------
// The same filter the on-orbit-capture instrument runs, headless. Tracks one
// axis of a noisy relative-position measurement stream.
export function kfMake() { return { x: 0, v: 0, P: [9, 0, 0, 120], init: false }; }
export function kfStep(k, z, dt, r) {
  if (!k.init) { k.x = z; k.v = 0; k.P = [r, 0, 0, 120]; k.init = true; return k; }
  k.x += k.v * dt;
  const qp = 0.4, qv = 30;
  const P00 = k.P[0] + dt * (k.P[2] + k.P[1]) + dt * dt * k.P[3] + qp;
  const P01 = k.P[1] + dt * k.P[3], P10 = k.P[2] + dt * k.P[3], P11 = k.P[3] + qv;
  const S = P00 + r, K0 = P00 / S, K1 = P10 / S, innov = z - k.x;
  k.x += K0 * innov; k.v += K1 * innov;
  k.P = [(1 - K0) * P00, (1 - K0) * P01, P10 - K1 * P00, P11 - K1 * P01];
  return k;
}
