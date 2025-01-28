# simulations/simulation.py
from rocketpy import Environment, SolidMotor, Rocket, Flight
from control_loops.control_loop import RocketController
import numpy as np
import time


def run_simulation():
    # Define the environment
    start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(f"\n[INFO] Simulation started at {start_time}")

    env = Environment(latitude=32.990254, longitude=-106.974998, elevation=0)
    env.set_atmospheric_model(
        type="custom_atmosphere", wind_u=[(0, 3), (10000, 3)], wind_v=[(0, 5), (10000, -5)]
        #type="custom_atmosphere", wind_u=[(0, 0), (10000, 0)], wind_v=[(0, 0), (10000, 0)]
    )
    
    # Define the rocket
    Pro75M1670 = SolidMotor(
        thrust_source="data/input/motors/Cesaroni_M1670.eng",
        dry_mass=1.815,
        dry_inertia=(0.125, 0.125, 0.002),
        nozzle_radius=33 / 1000,
        grain_number=5,
        grain_density=1815,
        grain_outer_radius=33 / 1000,
        grain_initial_inner_radius=15 / 1000,
        grain_initial_height=120 / 1000,
        grain_separation=5 / 1000,
        grains_center_of_mass_position=0.397,
        center_of_dry_mass_position=0.317,
        nozzle_position=0,
        burn_time=3.9,
        throat_radius=11 / 1000,
        coordinate_system_orientation="nozzle_to_combustion_chamber",
    )

    calisto = Rocket(
        radius=127 / 2000,
        mass=16.426,
        inertia=(6.321, 6.321, 0.034),
        power_off_drag="data/input/calisto/powerOffDragCurve.csv",
        power_on_drag="data/input/calisto/powerOnDragCurve.csv",
        center_of_mass_without_motor=0,
        coordinate_system_orientation="tail_to_nose",
    )

    rail_buttons = calisto.set_rail_buttons(
        upper_button_position=0.0818,
        lower_button_position=-0.618,
        angular_position=45,
    )

    calisto.add_motor(Pro75M1670, position=-1.255)

    nose_cone = calisto.add_nose(length=0.55829, kind="vonKarman", position=1.278)

    fin_set = calisto.add_trapezoidal_fins(
        n=4,
        root_chord=0.120,
        tip_chord=0.060,
        span=0.110,
        position=-1.04956,
        cant_angle=0.1,
        airfoil=("data/input/calisto/NACA0012-radians.csv", "radians"),
    )

    tail = calisto.add_tail(
        top_radius=0.0635, bottom_radius=0.0435, length=0.060, position=-1.194656
    )

     # Define controller with environmental parameters
    controller = RocketController(
        launch_latitude=env.latitude,
        launch_longitude=env.longitude,
        launch_elevation=env.elevation
    )

    air_brakes = calisto.add_air_brakes(
        drag_coefficient_curve="data/input/calisto/air_brakes_cd.csv",
        controller_function=controller.control_loop_function,  # Controller with Kalman observer
        sampling_rate=200,
        reference_area=None,
        clamp=True, 
        # Initialise the simulation ouput vector, [time, true_states, measurements, estimated_states]
        initial_observed_variables=[], # TODO fix this is a pain in the ass
        override_rocket_drag=False,
        name="AirBrakes",
        controller_name="AirBrakes Controller",
    )

    flight = Flight(
        rocket=calisto,
        environment=env,
        rail_length=10,
        inclination=90,
        heading=0,
        time_overshoot=False,
        terminate_on_apogee=True,
    )
    
    end_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(f"\n[INFO] Simulation ended at {end_time}")

    return flight
