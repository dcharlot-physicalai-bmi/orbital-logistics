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

// --- Rendezvous & proximity operations: Clohessy-Wiltshire relative motion ---
// Linearised relative motion of a chaser about a target on a circular orbit, in
// the target's LVLH frame (x = radial/out, y = along-track). Cross-track (z)
// decouples as a simple harmonic oscillator. This is the analytic backbone under
// the on-orbit-capture link: the estimator resolves the state, and CW says how a
// burn moves it. Mean motion n = sqrt(mu / a^3).
export function meanMotion(altitude_m) {
  const a = R_EARTH + altitude_m;
  return Math.sqrt(MU_EARTH / (a * a * a)); // rad/s
}

// The 2x2 planar CW state-transition blocks at elapsed time t. Returns rr, rv,
// vr, vv such that r(t) = rr*r0 + rv*v0 and v(t) = vr*r0 + vv*v0.
export function cwStm(n, t) {
  const s = Math.sin(n * t), c = Math.cos(n * t), nt = n * t;
  return {
    rr: [[4 - 3 * c, 0], [6 * (s - nt), 1]],
    rv: [[s / n, 2 * (1 - c) / n], [2 * (c - 1) / n, (4 * s - 3 * nt) / n]],
    vr: [[3 * n * s, 0], [6 * n * (c - 1), 0]],
    vv: [[c, 2 * s], [-2 * s, 4 * c - 3]],
  };
}
const mv = (M, v) => [M[0][0] * v[0] + M[0][1] * v[1], M[1][0] * v[0] + M[1][1] * v[1]];
const inv2 = (M) => { const d = M[0][0] * M[1][1] - M[0][1] * M[1][0];
  return [[M[1][1] / d, -M[0][1] / d], [-M[1][0] / d, M[0][0] / d]]; };

// Propagate a planar relative state (r0=[x,y] m, v0=[vx,vy] m/s) forward by t.
export function cwPropagate(r0, v0, n, t) {
  const P = cwStm(n, t);
  return {
    r: [mv(P.rr, r0)[0] + mv(P.rv, v0)[0], mv(P.rr, r0)[1] + mv(P.rv, v0)[1]],
    v: [mv(P.vr, r0)[0] + mv(P.vv, v0)[0], mv(P.vr, r0)[1] + mv(P.vv, v0)[1]],
  };
}

// Two-impulse targeting: from relative state (r0, v0now), find the departure
// velocity v0 that arrives at the origin (the target) after time t, and price
// both burns. dv1 injects onto the transfer; dv2 nulls the arrival velocity for
// a soft berth. Returns metres/second. This is the RPO maneuver the certified
// corridor is flown along.
export function cwTargetIntercept(r0, v0now, n, t) {
  const P = cwStm(n, t);
  // r(t)=0  =>  rv*v0 = -rr*r0  =>  v0 = -rv^{-1} (rr r0)
  const rrr = mv(P.rr, r0);
  const v0 = mv(inv2(P.rv), [-rrr[0], -rrr[1]]);
  const vArr = [mv(P.vr, r0)[0] + mv(P.vv, v0)[0], mv(P.vr, r0)[1] + mv(P.vv, v0)[1]];
  const dv1 = [v0[0] - v0now[0], v0[1] - v0now[1]];
  const dv2 = [-vArr[0], -vArr[1]];
  const mag = (u) => Math.hypot(u[0], u[1]);
  return { v0, vArr, dv1, dv2, dvTotal: mag(dv1) + mag(dv2) };
}

// --- MPPI: a receding-horizon optimal controller for the capture link ---------
// Model Predictive Path Integral control on the CW dynamics: sample many control
// sequences, roll each out, and take the cost-weighted average as the plan. Cost
// pulls the chaser to a soft berth on the target while holding an approach
// corridor (small along-track offset near the target) and spending little fuel.
// This is the on-device, optimal-control counterpart to the hand-tuned loop --
// the tube-MPC / chance-constrained-RL frontier, runnable in a browser.
// State s = [x, y, vx, vy] in the target's LVLH frame; control u = [ax, ay].
function mulberry32(a){ return function(){ a|=0; a=a+0x6D2B79F5|0; let t=Math.imul(a^a>>>15,1|a);
  t=t+Math.imul(t^t>>>7,61|t)^t; return ((t^t>>>14)>>>0)/4294967296; }; }
function gauss(rng){ // Box-Muller
  let u=0,v=0; while(u===0)u=rng(); while(v===0)v=rng();
  return Math.sqrt(-2*Math.log(u))*Math.cos(2*Math.PI*v); }

export function cwStepEuler(s, u, n, dt){
  const ax = 3*n*n*s[0] + 2*n*s[3] + u[0];
  const ay = -2*n*s[2] + u[1];
  return [s[0]+s[2]*dt, s[1]+s[3]*dt, s[2]+ax*dt, s[3]+ay*dt];
}
const MPPI_DEFAULTS = {
  K:220, H:28, dt:1.0, lambda:8, sigma:0.010, umax:0.030, // samples, horizon, s, temp, noise, max accel
  wPos:2.0, wVel:12.0, wCorr:6.0, wEffort:200, wOvershoot:40, corridorRange:12,
};
// Roll out one control sequence; return its total cost.
function mppiCost(s0, useq, n, o){
  let s=s0, c=0;
  for(let h=0; h<useq.length; h++){
    s = cwStepEuler(s, useq[h], n, o.dt);
    const rng2 = s[0]*s[0]+s[1]*s[1];
    c += o.wPos*rng2 + o.wVel*(s[2]*s[2]+s[3]*s[3]) + o.wEffort*(useq[h][0]**2+useq[h][1]**2);
    if(rng2 < o.corridorRange*o.corridorRange) c += o.wCorr*(s[1]*s[1]); // hold the corridor near the target
    if(s[0] < 0) c += o.wOvershoot*(s[0]*s[0]);                          // don't drive through the target
  }
  return c;
}
// One MPPI step: given the current state and warm-start nominal (H x 2), return
// the optimal nominal sequence (cost-weighted). First entry is the control to apply.
export function mppiPlan(s, nominal, n, opts={}, seed=1){
  const o={...MPPI_DEFAULTS, ...opts}, rng=mulberry32(seed);
  const H=o.H, samples=[], costs=[]; let beta=Infinity;
  for(let k=0;k<o.K;k++){
    const useq=new Array(H);
    for(let h=0;h<H;h++){
      let ax=nominal[h][0]+gauss(rng)*o.sigma, ay=nominal[h][1]+gauss(rng)*o.sigma;
      const m=Math.hypot(ax,ay); if(m>o.umax){ax*=o.umax/m;ay*=o.umax/m;}
      useq[h]=[ax,ay];
    }
    const c=mppiCost(s,useq,n,o); samples.push(useq); costs.push(c); if(c<beta)beta=c;
  }
  let wsum=0; const w=costs.map(c=>{const e=Math.exp(-(c-beta)/o.lambda); wsum+=e; return e;});
  const out=new Array(H);
  for(let h=0;h<H;h++){ let ax=0,ay=0; for(let k=0;k<o.K;k++){ax+=w[k]*samples[k][h][0]; ay+=w[k]*samples[k][h][1];}
    out[h]=[ax/wsum, ay/wsum]; }
  return out;
}
// --- 6-DOF: torque-free rigid-body tumble of a non-cooperative target ----------
// A dead satellite tumbles under no torque. Its attitude follows Euler's equations
// on the principal inertias I=[I1,I2,I3], integrated with the quaternion kinematics.
// This is the honest motion the pose estimator must track (not a fixed-axis spin):
// for I1<I2<I3 rotation about the intermediate axis is unstable (the tennis-racket
// theorem), so a real tumble wanders. Two quantities are invariant and let us check
// the integrator: the rotational kinetic energy and the angular-momentum magnitude.
export function eulerAngularAccel(w, I) {
  return [
    (I[1] - I[2]) / I[0] * w[1] * w[2],
    (I[2] - I[0]) / I[1] * w[2] * w[0],
    (I[0] - I[1]) / I[2] * w[0] * w[1],
  ];
}
// quaternion (w,x,y,z) times pure-vector angular rate -> qdot = 0.5 q ⊗ (0,ω)
function qMulVec(q, w) {
  const [qw, qx, qy, qz] = q, [wx, wy, wz] = w;
  return [
    0.5 * (-qx * wx - qy * wy - qz * wz),
    0.5 * (qw * wx + qy * wz - qz * wy),
    0.5 * (qw * wy - qx * wz + qz * wx),
    0.5 * (qw * wz + qx * wy - qy * wx),
  ];
}
export function rigidBodyStep(q, w, I, dt) {
  // RK2 (midpoint) on the coupled attitude + rate, then renormalize the quaternion
  const a1 = eulerAngularAccel(w, I), qd1 = qMulVec(q, w);
  const wm = w.map((v, i) => v + a1[i] * dt / 2);
  const qm = q.map((v, i) => v + qd1[i] * dt / 2);
  const a2 = eulerAngularAccel(wm, I), qd2 = qMulVec(qm, wm);
  const wn = w.map((v, i) => v + a2[i] * dt);
  let qn = q.map((v, i) => v + qd2[i] * dt);
  const n = Math.hypot(...qn) || 1; qn = qn.map(v => v / n);
  return { q: qn, w: wn };
}
export const rotEnergy = (w, I) => 0.5 * (I[0] * w[0] ** 2 + I[1] * w[1] ** 2 + I[2] * w[2] ** 2);
export const angMomentum = (w, I) => Math.hypot(I[0] * w[0], I[1] * w[1], I[2] * w[2]);
// rotate a body-frame point into the world by the quaternion
export function qRotate(q, p) {
  const [qw, qx, qy, qz] = q, [px, py, pz] = p;
  const tx = 2 * (qy * pz - qz * py), ty = 2 * (qz * px - qx * pz), tz = 2 * (qx * py - qy * px);
  return [px + qw * tx + qy * tz - qz * ty, py + qw * ty + qz * tx - qx * tz, pz + qw * tz + qx * ty - qy * tx];
}

// Closed-loop capture: run MPPI receding-horizon from s0 until berthed or timeout.
export function mppiCapture(s0, n, opts={}, seed=1){
  const o={...MPPI_DEFAULTS, ...opts};
  let s=s0.slice(), nominal=Array.from({length:o.H},()=>[0,0]), dv=0, captured=false;
  const traj=[s.slice()]; const maxSteps=opts.maxSteps||400;
  for(let step=0; step<maxSteps; step++){
    nominal = mppiPlan(s, nominal, n, o, seed+step);
    const u = nominal[0];
    s = cwStepEuler(s, u, n, o.dt);
    dv += Math.hypot(u[0],u[1])*o.dt;
    traj.push(s.slice());
    nominal = [...nominal.slice(1), [0,0]]; // shift the warm start
    const range=Math.hypot(s[0],s[1]), speed=Math.hypot(s[2],s[3]);
    if(range<0.5 && speed<0.05 && Math.abs(s[1])<0.4){ captured=true; break; }
  }
  return { captured, dvTotal:dv, steps:traj.length-1, trajectory:traj, finalRange:Math.hypot(s[0],s[1]) };
}
