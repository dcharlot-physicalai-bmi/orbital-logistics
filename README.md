<!-- Institute for Physical AI @ BMI · The Charlot Lab -->

# Orbital Logistics

**Physical AI for space logistics and transportation.**
Institute for Physical AI @ BMI · The Charlot Lab · Technical Report **TR-2026-17**

Reaching a destination in space is a velocity budget. The whole chain — launch, orbital transfer, rendezvous, capture, servicing, assembly, and return — becomes tractable when reusable, shared infrastructure and autonomous in-space transport pay that budget down. This repository is the **instrument suite** and the **headless physics core** behind it: each transport mode running on its real equations, and the autonomy that unites them, so the capabilities can be tested, applied, and deployed toward an open, decentralized space economy.

AI, Physical AI, and Embodied AI are opening new efficiencies, new materials, and new optimizations across space transportation and the management of space logistics. This track is the research into what those capabilities are and how to field them.

**Live topic & suite →** [physicalai-bmi.org/research/charlot-lab/space-logistics](https://physicalai-bmi.org/research/charlot-lab/space-logistics)

---

## The instruments

Each is a self-contained page that runs fully in the browser, on the device. Open `instruments/<name>.html`, tune it, and read where the gate is autonomy and control rather than materials.

| Instrument | File | What it is |
|---|---|---|
| **Compose the journey** | `instruments/compose.html` | A Δv-architecture composer: stack reusable infrastructure and autonomous in-space transport against a destination's velocity budget. The mass-ratio math is the exact rocket equation; a segment a single vehicle could never fly is marked so, and payload is read as a real mass fraction. |
| **Precision landing** | `instruments/landing.html` | Arriving at a surface. Closed-form **ZEM/ZEV** powered-descent guidance nulls position and velocity at touchdown under a hard thrust limit, picking a hazard-free pad; under-power it and it correctly cannot land soft. Moon / Mars / Earth. Toggle where the time-to-go comes from: **Scheduled** (search a flight time, fly the clock down), **Clock-free** (solve the optimal time-to-go from the state every step), or **Learned** (a released 1,281-param net predicts it in one forward pass, on-device). |
| **On-orbit capture** | `instruments/orbital-capture.html` | The autonomy that unites the chain. A chaser estimates a **tumbling, non-cooperative** target's pose with a Kalman filter, holds a certified approach corridor, matches the spin, and latches. Autonomous vs **MPPI** vs the **learned** on-device policy vs human-in-loop. |
| **Monocular pose** | `instruments/pose-vision.html` | The SPEED benchmark task: a target in full 6-DOF torque-free tumble, its pose recovered from one camera's noisy keypoints by perspective-n-point, scored live (~1–2° rotation, a few cm translation). |
| **On-orbit assembly** | `instruments/assembly.html` | Structures too large to launch whole, built in orbit: a free-flyer plans a valid build order, fetches each module, aligns to tolerance, and mates the joint — a boom, array, or ring truss, with no crew. |
| **ISRU · propellant** | `instruments/isru.html` | Make the fuel where you land: an autonomous excavator mines regolith, an electrolysis plant splits out its oxygen throttled to solar power (day/night), and a tank fills on the surface. Lunar or Mars, priced in kWh/kg. |
| **Autonomous traverse** | `instruments/mobility.html` | Hazard-aware rover navigation (AutoNav): potential-field steering around rocks and slopes toward each science target, drilling a sample, target after target — planned and driven on the vehicle. |
| **Formation flight** | `instruments/formation.html` | A self-coordinating constellation: each satellite holds its slot from its neighbours alone, avoids the others, and on a new pattern claims the nearest open slot and transfers collision-free. Distributed coordination, no central control — the formation keeps its own shape and reconfigures between ring, train, aperture, and wedge. |
| **Servicing & assembly** | `instruments/servicing.html` | Past capture: a manipulator inserts a replacement unit into a berthed satellite. Rigid position control jams on the chamfer; compliant force control lets the geometry seat it. Compliance, not precision. |
| **Skyhook · rotovator** | `instruments/skyhook.html` | The momentum-exchange tether on real orbital mechanics: catch/release velocities from vis-viva, the payload's transfer orbit, and the tether-to-payload mass from the constant-stress taper equation. |
| **Maglev sled** | `instruments/sled.html` | The electromagnetic ground assist: track length grows with the square of exit speed and is capped by survivable g-load. Reads track length, peak power, and coil-switching rate. |
| **Launch loop** | `instruments/launch-loop.html` | An elevated track held up by a fast internal rotor: an unstable equilibrium stabilized span by span. Ordinary steel; the stiffness is the controller. |

The instruments are the same bytes published at `physicalai-bmi.org/assets/sims/` — if you edit one here, update the site copy too.

## The core

The load-bearing equations, extracted from the instruments so they run headless for parameter sweeps. Nothing here is fitted or fabricated; every function is the same math the instrument draws.

```
core/physics.mjs        error function, tether taper, vis-viva, escape,
                        EM-launch track/power, Δv composer, capture Kalman filter,
                        Clohessy-Wiltshire relative motion + two-impulse RPO targeting,
                        MPPI receding-horizon optimal capture controller,
                        ZEM/ZEV powered-descent landing + clock-free optimal time-to-go,
                        torque-free 6-DOF rigid body,
                        decentralized formation flight (slot assignment + local keeping)
core/sweep.mjs          CLI: tether | sled | skyhook | compose | rpo | mppi
core/physics.test.mjs   sanity checks against independently re-derived values
```

```bash
npm test                    # 27 checks
npm run sweep -- tether     # tether mass ratio vs tip speed, per material
npm run sweep -- sled       # track length / peak power vs exit speed
npm run sweep -- skyhook    # catch/release vs altitude, against escape at the tip
npm run sweep -- compose    # a strong known architecture per destination
npm run sweep -- rpo        # two-impulse CW rendezvous cost vs transfer time
npm run sweep -- mppi       # MPPI receding-horizon capture from several standoffs
```

A few values the tests pin (all independently re-derived):

- **Maglev sled** — 2.9 km/s at 10 g needs **42.9 km** of track, ~14 GW peak, ~210 GJ for a 50 t vehicle.
- **Tether** — at a 3 km/s tip, the mass ratio is **13.6×** in Kevlar-49 and **0.26×** in carbon nanotube.
- **Skyhook** — a 600 km station with a 400 km arm releases a payload at **10.56 km/s**, above the **10.40 km/s** escape velocity *at the tip radius* (≈1000 km) — the comparison must be made at the tip, not the station.
- **Compose** — a single vehicle cannot reach Mars' surface (18 km/s) at all; composed with infrastructure it delivers a healthy payload mass fraction.
- **RPO** — the Clohessy-Wiltshire model reproduces the classics: two objects separated along-track on one circular orbit hold station, a radial offset drifts along-track, and a two-impulse targeting solve reaches the target and nulls its arrival velocity for a soft berth. This is the analytic backbone under the on-orbit-capture link.
- **MPPI** — a real Model Predictive Path Integral controller samples control sequences, rolls them out on the CW dynamics, and takes the cost-weighted average as the plan (receding-horizon). It berths a chaser from a standoff while holding the approach corridor at a bounded fuel cost, and is deterministic under a fixed seed. This is the optimal-control counterpart to a hand-tuned loop — the tube-MPC / chance-constrained-RL frontier, runnable on the device.

## What the topic argues

One shape repeats across every link in the chain: an estimator resolving state, and a controller holding a proven corridor. A launch handoff at speed, a momentum-exchange catch, a non-cooperative capture — each is that same problem, run on the vehicle at the edge. So a single autonomy stack, priced in Δv and joules, serves the whole logistics chain. That is the research this track develops, and the capability the space economy needs to open.

## License & provenance

MIT (`LICENSE`). Δv budgets are representative mission values; the mass-ratio, tether, orbital, and estimator math is exact and checked in `core/physics.test.mjs`. Illustrative dynamics are labelled as such inside each instrument.

---

*Institute for Physical AI @ BMI · The Charlot Lab. Part of the living corpus at [physicalai-bmi.org/research/charlot-lab](https://physicalai-bmi.org/research/charlot-lab).*

## RPO-Bench

An open, deterministic benchmark for the two hard links — non-cooperative **capture** (control) and monocular 6-DOF **pose** (perception). Fixed scenarios, fixed seeds, bit-identical every run, in Node or in a browser.

```bash
node bench/rpo_bench.mjs     # prints the leaderboard, writes bench/results.json
```

Baselines (reproducible):

| Capture (40 scenarios) | success | mean Δv | Pose (30 tracks) | rot err | SPEED |
|---|---|---|---|---|---|
| MPPI | 87.5% | 2.50 m/s | PnP (Gauss-Newton) | 1.27° | 0.028 |
| Learned (1,282-param) | 97.5% | **2.05 m/s** | Centroid (naive) | 23.22° | 0.417 |
| PD | 100% | 4.10 m/s | | | |

Live leaderboard (runs the fast baselines in your browser, bit-identical): `bench/rpo-bench.html` → [physicalai-bmi.org/assets/sims/rpo-bench](https://physicalai-bmi.org/assets/sims/rpo-bench).

## Landing-Bench

An open, deterministic benchmark for **autonomous powered descent**: 120 fixed starts (40 each on Moon / Mars / Earth) varying altitude, downrange offset, entry velocity and engine thrust-to-weight, scored on whether the vehicle landed soft (<2 m/s), on the pad (<8 m), and at what fuel cost.

```bash
node bench/landing_bench.mjs     # prints the table, writes bench/landing-results.json
```

| Guidance | success | mean miss | touchdown | mean Δv |
|---|---|---|---|---|
| Scheduled ZEM/ZEV | 95.0% | 0.03 m | 0.45 m/s | 139 m/s |
| Clock-free ZEM/ZEV | 90.8% | 0.00 m | 0.02 m/s | 159 m/s |
| Learned · thrust (1,346p) | 60.0% | 0.12 m | 1.05 m/s | 142 m/s |
| **Learned · tgo (1,281p)** | **91.7%** | **0.00 m** | **0.03 m/s** | 160 m/s |

Three things this makes concrete:

- **ZEM/ZEV does not need a clock.** For a fixed time-to-go the minimum-energy cost of the ZEM/ZEV solution is itself closed-form, `J(tgo) = 12|ZEM|²/tgo³ − 12(ZEM·ZEV)/tgo² + 4|ZEV|²/tgo`, so the optimal time-to-go can be solved from the *state* at every step (`optimalTgo`, `landGuidance`, `landFeedback`). The vehicle needs no plan and no schedule — and lands an order of magnitude more precisely (0.00 m / 0.02 m/s vs 0.03 m / 0.45 m/s).
- **The honest trade.** The scheduled search still wins on success rate and fuel: on thrust-marginal starts it can find *some* feasible flight time where the energy-optimal time-to-go demands more thrust than the engine has. Clock-free buys precision and autonomy for about 15% more Δv.
- **Factorization beat the training algorithm.** Cloning the *thrust* fails (9/120 with plain behavior cloning; 72/120 with 6 rounds of DAgger, and it plateaus) because the ZEM/ZEV gains go as 6/tgo² — the target is stiff exactly at touchdown. Cloning only the *time-to-go* and keeping the closed form reaches **110/120, at round 0, before any DAgger**. Learn the part that is expensive and smooth; keep the part that is cheap and exact.

```bash
node policy/train_land_tgo.mjs   # the shipped policy: state -> time-to-go (+ DAgger)
node policy/train_land.mjs       # baseline: clone the thrust directly (fails)
node policy/dagger_land.mjs      # baseline: clone the thrust + DAgger (plateaus)
```

Released: [physicalai-bmi/orbital-landing-tgo](https://huggingface.co/physicalai-bmi/orbital-landing-tgo).

## Formation-Bench

The multi-agent counterpart: an open, deterministic benchmark for **decentralized formation flight**. Twelve agents hold a rotating formation and reconfigure it (ring→grid→wedge→line) with local control only, scored on whether the shape holds (RMS slot error), whether it stays collision-free (global minimum separation vs the hard-body radius), and the Δv it costs — 30 fixed seeds, bit-identical every run.

```bash
node bench/formation_bench.mjs     # prints the table, writes bench/formation-results.json
```

Two controllers (reproducible):

| Controller | success | mean RMS | worst min-sep | mean Δv/agent |
|---|---|---|---|---|
| Distributed (analytic) | 100% | 0.09 m | 12.11 m | 303 m/s |
| Learned (1,474-param, on-device) | 100% | 0.15 m | 13.05 m | 302 m/s |

Two findings the benchmark makes concrete:

- **The assignment is the safety-critical step, not the controller.** The dominant collision risk is a *crossing transfer* — two agents swapping across the formation. A minimum sum-of-squared-distance matching is provably non-crossing, and a 2-opt untangling pass (`assignSlots`) converges to it; with crossings removed, every controller reconfigures collision-free with a 12–13 m margin. Naive greedy assignment, by contrast, leaves near-misses under 1.5 m.
- **A compact learned policy reproduces the coordination.** A 1,474-parameter per-agent net — behavior-cloned from the analytic controller, seeing only its own slot error and its three nearest neighbours (`policy/`) — holds the formation and reconfigures on-device, matching the analytic's keeping and Δv, under a light reflexive safety filter. Train it end to end:

```bash
node policy/gen_formation_data.mjs   # roll out the analytic controller → demonstrations
node policy/train_formation.mjs      # clone into formation_policy.json + closed-loop validate
```
