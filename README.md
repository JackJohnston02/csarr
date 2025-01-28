# CSAR - Control Schemes for Apogee Regulation in Rockets

Apogee regulation in high-powered rocketry is an increasingly popular goal for student rocketry competi-
tions. Traditional approaches to apogee regulation often depend on pre-computed trajectories or complex
offline characterisation simulations. This work presents the development of an adaptive super-twisting
sliding mode controller with the sliding surface approximated by a neural network, designed to operate in
real-time and without prior knowledge of the rocket’s parameters. An Extended Kalman Filter provides
real-time state and ballistic coefficient estimates, while a neural network evaluates the computationally
expensive sliding surface. Simulations using RocketPy validate the controller’s performance, highlighting
improvements in apogee accuracy, controller effort and computational efficiency when compared to exist-
ing methods. These findings establish the neural network super twisting controller as an effective, adaptive
solution to the apogee regulation problem, offering the potential for future real-world application.
*** 

## Repository Overview
```
├── diagrams
│   ├── airbrake_drag_coeff
│   └── regulus_drag
├── results
│   └── controller_comparison
│       ├── Apogee Prediction
│       ├── Reference Trajectory Controller
│       ├── input
│       │   ├── Dead-Band
│       │   ├── NN Super-Twisting
│       │   ├── Soft-Switching
│       │   └── Super-Twisting
│       └── output
│           └── plots
├── simulations
│   └── rocketPy
│       ├── analysis
│       │   ├── input
│       │   │   ├── Apogee Prediction Controller
│       │   │   ├── Controller Comparison
│       │   │   ├── Dead-Band Sliding Mode
│       │   │   ├── Neural Netowork Sliding Surface
│       │   │   ├── No Anti-Chattering Sliding Mode
│       │   │   ├── Real Flight Data
│       │   │   ├── Reference Trajectory Controller
│       │   │   ├── Soft-Switching Sliding Mode
│       │   │   └── Super-Twisting Sliding Mode
│       │   └── output
│       │       ├── Apogee Prediction Controller
│       │       ├── Controller Comparison
│       │       ├── Dead-Band Sliding Mode
│       │       ├── Neural Netowork Sliding Surface
│       │       ├── No Anti-Chattering Sliding Mode
│       │       ├── Real Flight Data
│       │       ├── Reference Trajectory Controller
│       │       ├── Soft-Switching Sliding Mode
│       │       └── Super-Twisting Sliding Mode
│       ├── control_loops
│       │   ├── __pycache__
│       │   ├── controllers
│       │   │   └── __pycache__
│       │   ├── estimators
│       │   │   └── __pycache__
│       │   ├── plants
│       │   └── sensors
│       │       └── __pycache__
│       ├── data
│       │   ├── input
│       │   │   ├── calisto
│       │   │   ├── motors
│       │   │   └── regulus
│       │   ├── output
│       │   │   └── __pycache__
│       │   └── target_manifolds
│       ├── plots
│       │   ├── comparison_states
│       │   ├── controller_states
│       │   ├── estimated_states
│       │   ├── measurements
│       │   ├── true_states
│       │   └── vertical_states
│       ├── plotting
│       │   └── __pycache__
│       ├── simulations
│       │   └── __pycache__
│       ├── test
│       └── utils
│           └── __pycache__
└── tests
    ├── backwards_propagation
    ├── neural_network
    │   ├── output
    │   └── utils
    │       └── __pycache__
    └── target_manifold
        ├── __pycache__
        └── output
```
