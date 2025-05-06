using Random, LinearAlgebra, StatsBase, DataFrames, CSV, Printf

cd(@__DIR__)

#This function implements a julia version of the original python code used to create data, again found at Chris Conlon's github. 
#This should be about an order of magnitude faster.

# RustModel definition — guards so I can run again and again, and again
if !isdefined(Main, :RustModel)
    mutable struct RustModel
        n_grid::Int
        x_values::Vector{Float64}
        beta::Float64
        params::Vector{Float64}
        cost_param::Vector{Float64}
        RC::Float64
        probs::Vector{Float64}
        tpm::Matrix{Float64}
        per_period_cost::Vector{Float64}
        EV::Vector{Float64}
        p_continue::Vector{Float64}
    end
end    

# Constructor and model setup
function RustModel(x_min::Int=0, x_max::Int=200, n_grid::Int=201, beta::Float64=0.975)
    x_vals = range(x_min/100, x_max/100, length=n_grid) |> collect
    tpm = Matrix{Float64}(I, n_grid, n_grid)
    EV = zeros(n_grid)
    per_cost = zeros(n_grid)
    p_cont = zeros(n_grid)
    return RustModel(n_grid, x_vals, beta, zeros(3), zeros(2), 0.0, [1.0,0.0,0.0], tpm, per_cost, EV, p_cont)
end

function set_parameters!(m::RustModel, params::Vector{Float64})
    @assert length(params)==3
    m.params, m.cost_param, m.RC = params, params[1:2], params[3]
    m.per_period_cost .= m.cost_param[1] .* m.x_values .+ m.cost_param[2] .* m.x_values.^2
end

function update_transition_probs!(m::RustModel, probs::Vector{Float64})
    n = m.n_grid; A = zeros(n,n)
    for (k,p) in enumerate(probs)
        offs = k-1
        for i in 1:n
            j = i+offs
            if j ≤ n
                A[i,j] = p
            end
        end
    end
    row_s = sum(A, dims=2)
    for i in 1:n
        A[i,end] += 1 - row_s[i]
    end
    m.tpm .= A
    m.probs = copy(probs)
end

function solve_value_function!(m::RustModel; tol=1e-8, maxiter=10_000)
    EV = copy(m.EV)
    for _ in 1:maxiter
        rep = m.beta * EV[1] - m.per_period_cost[1] - m.RC
        wait = m.beta .* EV .- m.per_period_cost
        cont = log.(exp(rep) .+ exp.(wait))
        EV_new = m.tpm * cont
        if maximum(abs.(EV_new .- EV)) < tol
            m.EV .= EV_new
            return
        end
        EV .= EV_new
    end
    error("Value function did not converge")
end

function compute_choice_probabilities!(m::RustModel)
    ex = -m.per_period_cost .+ m.beta .* m.EV
    vd = -ex .- m.RC .+ ex[1]
    m.p_continue .= 1 ./(1 .+ exp.(vd))
end

# Efficient bulk simulation avoiding DataFrame in loop
function simulate_buses(m::RustModel, n_buses::Int, n_periods::Int)
    total = n_buses * n_periods
    period_id = Vector{Int}(undef, total)
    bus_id    = Vector{Int}(undef, total)
    y_it      = Vector{Int}(undef, total)
    x_it      = Vector{Int}(undef, total)
    tmp_x = zeros(Int, n_periods)
    tmp_y = zeros(Int, n_periods)
    idx = 1
    for b in 1:n_buses
        u1   = rand(n_periods)
        incr = sample(0:length(m.probs)-1, Weights(m.probs), n_periods)
        cnt = 0
        for t in 1:n_periods
            cnt = min(cnt + incr[t], m.n_grid-1)
            if m.p_continue[cnt+1] < u1[t]
                tmp_y[t] = 1
                cnt = incr[t]
            else
                tmp_y[t] = 0
            end
            tmp_x[t] = cnt
        end
        for t in 1:n_periods
            period_id[idx] = t
            bus_id[idx]    = b
            y_it[idx]      = tmp_y[t]
            x_it[idx]      = tmp_x[t]
            idx += 1
        end
    end
    return period_id, bus_id, y_it, x_it
end

# Main execution
function main()
    n_buses, n_periods = 100_000, 150
    m = RustModel()
    set_parameters!(m, [2.4569, 0.03, 11.7257]) 
    update_transition_probs!(m, [0.0937, 0.4475, 0.4459, 0.0127, 0.0002]) #Only allow 5 states for now. By doing this I save time with transition probs. 
    solve_value_function!(m)
    compute_choice_probabilities!(m)

    # simulate and write full panel
    pid, bid, y, x = simulate_buses(m, n_buses, n_periods)
    CSV.write("rust_data_2025.csv", DataFrame(period_id=pid, bus_id=bid, y_it=y, x_it=x))

    # summarize replacements
    total_repl = sum(y)
    @printf("Number of replacements: %d\n\n", total_repl)

    # compute Δx; keep only 0 ≤ Δx ≤ 4- which is what our data makes anyway
    dx     = diff(x)
    dx_pos = dx[(dx .>= 0) .& (dx .<= 4)]

    # Pure-Julia histogram approach rather than DataFrames- saves time on the margin for big number of buses
    hist  = fit(Histogram, dx_pos, 0:5)
    freqs = hist.weights ./ sum(hist.weights)

    println("Transition frequencies (dx = 0 to 4):")
    for i in 0:4
        @printf("  Δx = %d → freq = %.6f\n", i, freqs[i+1])
    end

    @printf("\nSum of transition frequencies: %.4f\n", sum(freqs))
end