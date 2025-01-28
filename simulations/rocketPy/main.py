import time
# Import the simulation function
from simulations.simulation import run_simulation
from plotting.processing import process_flight_data, export_flight_data
from plotting.plot_measurements import plot_measurements
from plotting.plot_states import plot_true_states, plot_estimated_states, plot_comparison_states, plot_controller_states, plot_vertical_states

if __name__ == "__main__":
    start_time = time.time()
    # Run simulation    
    flight = run_simulation()
    end_time = time.time()

    print(f"Time to run: {end_time - start_time}")

    print(f"[INFO] Apogee altitude: {flight.apogee:}")
    
    # Process the flight datapyt
    data = process_flight_data(flight)

    # Plot the data
    plot_controller_states(data['times'], data['controller_states'])
    plot_measurements(data)
    plot_true_states(data['times'], data['true_states'])
    plot_estimated_states(data['times'], data['estimated_states'])
    plot_vertical_states(data['times'], data['estimated_states'], data['true_states'])
    plot_comparison_states(data['times'], data['true_states'], data['estimated_states'])
    
    # Generate a unique filename with a timestamp
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"data/output/flight_data_{timestamp}.csv"
    
    # Export the flight data to the unique filename
    export_flight_data(data, filename=filename)

    print(f"[INFO] Data plotted\n")
