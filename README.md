# Rust Estimation Project in Julia

## Overview

This repository implements the structural estimation of the bus-engine replacement model originally proposed by Rust (1987) using both the Nested Fixed Point (NFXP) and MPEC approaches, in Julia. For information about the problem set- visit https://github.com/chrisconlon/Grad-IO/tree/master/Assignments/Assignment%204-%20Rust.

## Repository Structure

- `IO_Rust_Julia.ipynb`  
  A Jupyter notebook containing all data preprocessing, estimation routines, and result visualizations.
- `IO_Rust_Julia.jl`  
  A Julia Script. It is not very optimized (hopefully soon), as it just takes and simplifies the jupyter notebook. 
- `Rust_Combined.jl`  
An updated and efficient Julia script which both generates data, and estimates model parameters.
- `rust_data_2020.csv`  
- `Rust_Monte_Carlo.jl`  
A script to perform a Monte Carlo Simulation, the output of which is the figures below. 
  Dataset with fields:
  - `bus_id`: Bus identifier  
  - `period_id`: Time period  
  - `y_it`: Replacement decision (1 = replace, 0 = keep)  
  - `x_it`: Mileage state  
- `Rust_Problems.pdf`  
  Problem set.
- `generate_data.jl`  
  Generates bus data

## Results

The notebook outputs parameter estimates, computation times, and EV function vizualizations for both NFXP and MPEC. I use JuMP to solve this problem; Chris used AMPL. 

From Monte Carlo: 

![alt text](https://github.com/Mtsoukis/Rust_Estimation-Julia/blob/main/Figures/theta1.png)
![alt text](https://github.com/Mtsoukis/Rust_Estimation-Julia/blob/main/Figures/theta2.png)
![alt text](https://github.com/Mtsoukis/Rust_Estimation-Julia/blob/main/Figures/RC.png)

## License

This project is released under the MIT License.

## Author

**Marios Tsoukis**

