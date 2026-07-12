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
| **On-orbit capture** | `instruments/orbital-capture.html` | The autonomy that unites the chain. A chaser estimates a **tumbling, non-cooperative** target's pose with a Kalman filter, holds a certified approach corridor, matches the spin, and latches. Autonomous vs human-in-loop. |
| **Skyhook · rotovator** | `instruments/skyhook.html` | The momentum-exchange tether on real orbital mechanics: catch/release velocities from vis-viva, the payload's transfer orbit, and the tether-to-payload mass from the constant-stress taper equation. |
| **Maglev sled** | `instruments/sled.html` | The electromagnetic ground assist: track length grows with the square of exit speed and is capped by survivable g-load. Reads track length, peak power, and coil-switching rate. |
| **Launch loop** | `instruments/launch-loop.html` | An elevated track held up by a fast internal rotor: an unstable equilibrium stabilized span by span. Ordinary steel; the stiffness is the controller. |

The instruments are the same bytes published at `physicalai-bmi.org/assets/sims/` — if you edit one here, update the site copy too.

## The core

The load-bearing equations, extracted from the instruments so they run headless for parameter sweeps. Nothing here is fitted or fabricated; every function is the same math the instrument draws.

```
core/physics.mjs        error function, tether taper, vis-viva, escape,
                        EM-launch track/power, Δv composer, capture Kalman filter
core/sweep.mjs          CLI: tether | sled | skyhook | compose
core/physics.test.mjs   sanity checks against independently re-derived values
```

```bash
npm test                    # 7 checks
npm run sweep -- tether     # tether mass ratio vs tip speed, per material
npm run sweep -- sled       # track length / peak power vs exit speed
npm run sweep -- skyhook    # catch/release vs altitude, against escape at the tip
npm run sweep -- compose    # a strong known architecture per destination
```

A few values the tests pin (all independently re-derived):

- **Maglev sled** — 2.9 km/s at 10 g needs **42.9 km** of track, ~14 GW peak, ~210 GJ for a 50 t vehicle.
- **Tether** — at a 3 km/s tip, the mass ratio is **13.6×** in Kevlar-49 and **0.26×** in carbon nanotube.
- **Skyhook** — a 600 km station with a 400 km arm releases a payload at **10.56 km/s**, above the **10.40 km/s** escape velocity *at the tip radius* (≈1000 km) — the comparison must be made at the tip, not the station.
- **Compose** — a single vehicle cannot reach Mars' surface (18 km/s) at all; composed with infrastructure it delivers a healthy payload mass fraction.

## What the topic argues

One shape repeats across every link in the chain: an estimator resolving state, and a controller holding a proven corridor. A launch handoff at speed, a momentum-exchange catch, a non-cooperative capture — each is that same problem, run on the vehicle at the edge. So a single autonomy stack, priced in Δv and joules, serves the whole logistics chain. That is the research this track develops, and the capability the space economy needs to open.

## License & provenance

MIT (`LICENSE`). Δv budgets are representative mission values; the mass-ratio, tether, orbital, and estimator math is exact and checked in `core/physics.test.mjs`. Illustrative dynamics are labelled as such inside each instrument.

---

*Institute for Physical AI @ BMI · The Charlot Lab. Part of the living corpus at [physicalai-bmi.org/research/charlot-lab](https://physicalai-bmi.org/research/charlot-lab).*
