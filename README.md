# PhysicsSimulator
This project develops a research-grade quadrotor physics simulator, integrating advanced geometric mechanics, fault-tolerant estimation, and mission-level safety verification. The simulator operates on SE(3) using Lie-group variational integrators (LGVI).


**Lie-Group Quadrotor Simulator (Stages 1–6)**


**Overview**

This repository implements a research-grade quadrotor simulator across 6 progressive development stages, integrating physics, estimation, control, and fault-tolerant mission management. The simulator leverages Lie-group variational integrators (LGVI) for SE(3) dynamics, a right-invariant EKF for state estimation, and safety envelopes for robust, certifiable UAV operations.

It is suitable for:

Academic research in geometric control and estimation

Safety-critical UAV simulations

Testing robustness under sensor faults, actuator failures, and environmental disturbances

**Project Stages**
Stage	Key Features
Stage 1:	Basic rigid-body quadrotor physics with translational/rotational dynamics. (Theoretical)

Stage 2:	Motor model with first-order lag, saturation, and simple thrust allocation. (Theoretical)

Stage 3:	Observer-in-the-loop: EKF-based position, velocity, and orientation estimation with noise.

Stage 4:	Monte Carlo falsification atlas for state estimation and trajectory verification.

Stage 5:	Certified geometric SE(3) control, Right-Invariant EKF, energy-based flight certification, and GPS failure analysis.

Stage 6:	NASA-grade fault-tolerant mission manager with adversarial scenario simulation, motor faults, GPS jamming, wind disturbances, and structured safety envelopes.

**Core Components**
1. Physics Engine (LGVI)

Preserves rotational dynamics on SO(3) using exponential maps.

Supports rigid-body gyroscopic effects, gravity, aerodynamic drag, and wind disturbances.

Integrates propulsion dynamics, including actuator lag, saturation, and fault injection.

2. Estimator

Right-Invariant Extended Kalman Filter (RI-EKF) on SE_2(3).

Estimates position, velocity, orientation, and sensor biases.

Tracks innovation metrics, covariance growth, whiteness, and condition number for estimator credibility.

3. Controller

Geometric SE(3) control for hover and trajectory tracking.

Fault-aware gain scheduling based on mission mode (Nominal, Degraded, Emergency).

Handles emergency descent under extreme faults or imminent collision.

4. Safety Envelope

Computes impact margin and energy envelope for preemptive safety actions.

Verifies flight viability under faults, wind, and sensor failures.

5. Mission Manager

Multi-mode state machine: Nominal → Degraded → Emergency.

Dynamically adjusts control gains and safety protocols.

Detects estimator divergence and triggers emergency protocols.

6. Falsification Campaign

Structured testing under adversarial scenarios:

Rotor failure / efficiency loss

GPS jamming

Wind shear disturbances

Combined “perfect storm” conditions

Provides trajectory, error, and safety margin logs for system certification.
