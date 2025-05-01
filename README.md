# Rust Estimation Project in Julia

## Overview

This repository implements the structural estimation of the bus-engine replacement model originally proposed by Rust (1987) using both the Nested Fixed Point (NFXP) and MPEC approaches, in Julia. For information about the problem set- visit https://github.com/chrisconlon/Grad-IO/tree/master/Assignments/Assignment%204-%20Rust.

## Repository Structure

- `IO_Rust_Julia.ipynb`  
  A Jupyter notebook containing all data preprocessing, estimation routines, and result visualizations.
- `IO_Rust_Julia.jl`  
  A Julia Script. It is not very optimized (hopefully soon), as it just takes and simplifies the jupyter notebook. 
- `rust_data_2020.csv`  
  Dataset with fields:
  - `bus_id`: Bus identifier  
  - `period_id`: Time period  
  - `y_it`: Replacement decision (1 = replace, 0 = keep)  
  - `x_it`: Mileage state  
- `Rust_Problems.pdf`  
  Problem set.
- `README.md`  
  This file.

## Results

The notebook outputs parameter estimates, computation times, and EV function visualizations for both NFXP and MPEC. I show that using JuMP indeed does solve this problem; Chris used AMPL. 

## License

This project is released under the MIT License.

## Author

**Marios Tsoukis**

