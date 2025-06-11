using Random, LinearAlgebra, StatsBase, Printf, BenchmarkTools
using JuMP, Ipopt, Optim, ForwardDiff, SparseArrays

# ------------------------------
#   constants & grids
# ------------------------------
const N      = 201          # size of mileage grid
const δ      = 0.975        # discount factor 
const XGRID  = collect(0:N-1) .* 0.01
const XGRID2 = XGRID .^ 2

# ------------------------------------------------------
#  Structural model object & sub‑routines
# ------------------------------------------------------
mutable struct RustModel{T<:AbstractFloat,M<:AbstractMatrix{T}}
    n_grid::Int
    x_values::Vector{T}
    beta::T
    params::Vector{T}
    cost_param::Vector{T}
    RC::T
    probs::Vector{T}
    tpm::M
    per_period_cost::Vector{T}
    EV::Vector{T}
    p_continue::Vector{T}
end

function RustModel(x_min::Int=0, x_max::Int=200, n_grid::Int=N, beta::Float64=δ)
    x_vals = collect(range(x_min/100, x_max/100, length=n_grid))
    RustModel(n_grid,
              x_vals,
              beta,
              zeros(3),                # θ placeholder
              zeros(2),                # θ cost
              0.0,                     # RC
              Float64[],               # probs initialised empty; set later
              Matrix{Float64}(I, n_grid, n_grid),
              zeros(n_grid),
              zeros(n_grid),
              zeros(n_grid))
end

function set_parameters!(m::RustModel, params::Vector{Float64})
    @assert length(params)==3
    m.params      .= params
    m.cost_param  .= params[1:2]
    m.RC           = params[3]
    @. m.per_period_cost = m.cost_param[1]*m.x_values + m.cost_param[2]*m.x_values^2
end

function update_transition_probs!(m::RustModel, probs::Vector{Float64})
    n             = m.n_grid
    num_increments = length(probs)

    #  Count non-zeros in TPM 
    nnz = 0
    @inbounds for k in 1:num_increments          # increment index
        p = probs[k]
        if p > 0.0
            inc = k - 1                           # mileage jump
            nnz += n - inc                       
        end
    end

    #  Pre-allocate the I/J/V arrays once
    I = Vector{Int}(undef, nnz)                  # row indices
    J = similar(I)                               # col indices
    V = Vector{Float64}(undef, nnz)              # probabilities

    idx = 1
    @inbounds for (inc_idx, p) in enumerate(probs)
        p > 0.0 || continue
        inc   = inc_idx - 1                 
        last  = n - inc
        for row in 1:last
            I[idx] = row
            J[idx] = row + inc                  
            V[idx] = p
            idx   += 1
        end
    end

    # sparse matrix
    P_temp = sparse(I, J, V, n, n)      

    row_sums = sum(P_temp, dims = 2)
    @inbounds for r in 1:n
        rem = 1.0 - row_sums[r]
   
    end

    m.tpm   = P_temp
    m.probs = copy(probs)
    return nothing
end


function solve_value_function!(m::RustModel; tol=1e-8, maxiter=10_000)
    EV_old = copy(m.EV) 
    
    # Pre-allocate vectors for efficiency
    EV_new = similar(m.EV)
    expected_future_val_if_wait = similar(m.EV)
    val_wait_state_specific = similar(m.EV)

    for _ in 1:maxiter
        val_replace = -m.RC + m.beta * dot(view(m.tpm,1,:), EV_old)
        mul!(expected_future_val_if_wait, m.tpm, EV_old)
        
        @. val_wait_state_specific = -m.per_period_cost + m.beta * expected_future_val_if_wait
        for j in 1:m.n_grid
            EV_new[j] = _stable_log_sum_exp(val_wait_state_specific[j], val_replace)
        end

        # Check for convergence using sup-norm- gracias Efe.
        if maximum(abs.(EV_new.- EV_old)) < tol
            m.EV.= EV_new #
            return
        end

        EV_old.= EV_new
    end
    error("Value function did not converge in $maxiter iterations")
end

function compute_choice_probabilities!(m::RustModel)
    
    expected_future_val_if_wait = m.tpm * m.EV 
    val_wait_state_specific =.-m.per_period_cost.+ m.beta.* expected_future_val_if_wait #

    val_replace = -m.RC + m.beta * dot(view(m.tpm,1,:), m.EV)

    exp_val_wait = exp.(val_wait_state_specific) 
    exp_val_replace_scalar = exp(val_replace)  
    
    m.p_continue.= exp_val_wait./ (exp_val_wait.+ exp_val_replace_scalar)
end

function _stable_log_sum_exp(a::T, b::T) where {T<:Real}
    max_val = max(a, b)
   
    if isinf(ForwardDiff.value(max_val)) && ForwardDiff.value(max_val) < 0.0
        return T(-Inf) # Ensure we return the same type T
    end
    return max_val + log(exp(a - max_val) + exp(b - max_val))
end

# ------------------------------------------------------
#  Simulation 
# ------------------------------------------------------

function simulate_buses(m::RustModel, n_buses::Int, n_periods::Int)
    total  = n_buses * n_periods
    period = Vector{Int}(undef, total)
    bus_id = Vector{Int}(undef, total)
    y_it   = Vector{Int}(undef, total)
    x_it   = Vector{Int}(undef, total)

    idx = 1
    for b in 1:n_buses
        u1   = rand(n_periods)       
        incr = sample(0:length(m.probs)-1,
                      Weights(m.probs),
                      n_periods)      
        cnt = 0

        @inbounds for t in 1:n_periods
            # Decision based on start-of-period mileage
            replace = (m.p_continue[cnt + 1] < u1[t])
            y_it[idx] = replace

            # If replace, reset mileage to zero for next period
            if replace
                cnt = 0
            end

            # Using the pre-drawn mileage increments:
            inc = incr[t]
            cnt = min(cnt + inc, m.n_grid - 1)
            x_it[idx] = cnt

            bus_id[idx] = b
            period[idx] = t
            idx += 1
        end
    end
    return period, bus_id, y_it, x_it
end

# ------------------------------------------------------
# Sufficient statistics
# ------------------------------------------------------
function build_transition_and_counts(bus_id::Vector{Int},
    x_it::Vector{Int},   
    y_it::Vector{Int},  
    N_states::Int)     

counts_keep = zeros(Int, N_states)
counts_rep = zeros(Int, N_states)
universal_increment_counts = zeros(Int, N_states)

#count decisions and transitions

for idx in eachindex(bus_id)
b = bus_id[idx]
current_x_end_period_0_idx = x_it[idx]
decision_in_current_period = y_it[idx]

mileage_at_start_of_current_period_0_idx = 0
if idx > 1 && bus_id[idx-1] == b
mileage_at_start_of_current_period_0_idx = x_it[idx-1]
end # At start, mileage = 0 (or rather at t = -1)

# Decision
if 0 <= mileage_at_start_of_current_period_0_idx < N_states
state_for_decision_1_idx = mileage_at_start_of_current_period_0_idx + 1
if decision_in_current_period == 0 # Kept
counts_keep[state_for_decision_1_idx] += 1
else # Replaced
counts_rep[state_for_decision_1_idx] += 1
end
end

# Calculate increment
actual_increment_0_idx = -1
if decision_in_current_period == 0 
actual_increment_0_idx = current_x_end_period_0_idx - mileage_at_start_of_current_period_0_idx
else 
actual_increment_0_idx = current_x_end_period_0_idx
end

# Store increment
if 0 <= actual_increment_0_idx < N_states
universal_increment_counts[actual_increment_0_idx + 1] += 1
end
end

#PMF 

P_universal_increment_pmf = zeros(Float64, N_states)
total_observed_increments = sum(universal_increment_counts)
if total_observed_increments > 0
P_universal_increment_pmf .= universal_increment_counts ./ total_observed_increments
elseif N_states > 0
# Fallback: if no increment- increment of 0 with prob 1.
P_universal_increment_pmf[1] = 1.0
end

#  Transition matrix using PMF

diagonals = [
    (d - 1) => fill(p, N_states - (d - 1))
    for (d, p) in enumerate(P_universal_increment_pmf) if p > 0.0
]
    
P_output = spdiagm(N_states, N_states, diagonals...)

row_sums = sum(P_output, dims=2)
P_output[:, N_states] .+= (1.0 .- vec(row_sums))

return P_output, counts_keep, counts_rep, universal_increment_counts
end
# ------------------------------------------------------
#  Estimation helpers 
# ------------------------------------------------------

#NFXP
function solve_ev!(V::AbstractVector{T}, P::AbstractMatrix,
                   θ₁::T, θ₂::T, RC::T, δ::T;
                   tol::T=T(1e-8), maxiter::Int=10_000) where {T}
    N  = length(V)
    EV = similar(V); v_k = similar(V)
    for _ in 1:maxiter
        mul!(EV,P,V)                                  # EV := P·V
        @inbounds @simd for j in 1:N                 
            xj    = XGRID[j]
            v_k[j]= -θ₁*xj - θ₂*xj^2 + δ*EV[j]
        end
        v_r = -RC + δ*dot(view(P,1,:),V)              
        diff = zero(T)
        @inbounds @simd for j in 1:N
            Vnew = _stable_log_sum_exp(v_k[j], v_r)
            d    = abs(Vnew - V[j])
            diff = ifelse(d>diff,d,diff)
            V[j] = Vnew
        end
        diff < tol && return
    end
    error("EV failed to converge in $maxiter iterations")
end

solve_ev(P, θ₁, θ₂, RC, δ, ::Type{T}=Float64) where {T} = (V=zeros(T,N); solve_ev!(V, T.(P), T(θ₁),T(θ₂),T(RC),T(δ)); V)

function ccp(P, V, θ₁, θ₂, RC, δ)
    EV   = P*V
    v_k  = @. -(θ₁*XGRID + θ₂*XGRID2) + δ*EV
    v_r  = -RC + δ*dot(P[1,:],V)
    exp_vr = exp(v_r)
    @. exp(v_k) / (exp(v_k) + exp_vr)
end

#MPEC
function estimate_mpec(P, ck, cr, δ)
    model = Model(Ipopt.Optimizer);
    set_silent(model)
    @variable(model, θ₁ ≥ 1e-6, start=1.0)
    @variable(model, θ₂ ≥ 1e-6, start=1.0)
    @variable(model, RC   ≥ 1e-6, start=1.0)
    @variable(model, V[1:N],     start=-5.0)
    @variable(model, v_k[1:N],   start=-5.0)
    @variable(model, v_r,        start=-15.0)

    @NLexpression(model, EV_keep[j=1:N], sum(P[j,k]*V[k] for k=1:N))
    @NLexpression(model, EV_rep,         sum(P[1,k]*V[k] for k=1:N))

    @NLconstraint(model, [j=1:N], v_k[j] + θ₁*XGRID[j] + θ₂*XGRID2[j] - δ*EV_keep[j] == 0)
    @NLconstraint(model, v_r + RC - δ*EV_rep == 0)
    @NLexpression(model, denom[j=1:N], exp(v_k[j]) + exp(v_r))
    @NLconstraint(model, [j=1:N], V[j] == log(denom[j]))

    @NLobjective(model, Min,
        -sum(ck[j]*log(exp(v_k[j])/denom[j]) + cr[j]*log(exp(v_r)/denom[j]) for j=1:N))

    set_optimizer_attribute(model, "tol", 1e-8)
    optimize!(model)
    return value(θ₁), value(θ₂), value(RC), -objective_value(model)
end

# Global containers for ForwardDiff’s gradient
P_glob  = spzeros(Float64, N, N)
ck_glob = zeros(Int,      N)
cr_glob = zeros(Int,      N)
dx_glob = zeros(Int,      N)

#Speed things up MPEC- gradient & LL 
function ll_and_grad!(F,G,θ::Vector{T}) where {T}
    θ₁, θ₂, RC = θ
    if θ₁≤0 || θ₂≤0 || RC≤0
        F[] = T(Inf)
        G !== nothing && fill!(G,zero(T))
        return
    end
    V = solve_ev(P_glob, θ₁, θ₂, RC, δ, typeof(θ₁))
    σ = ccp(P_glob, V, θ₁, θ₂, RC, δ)
    F[] = -sum(ck_glob .* log.(σ) .+ cr_glob .* log.(1 .- σ))
    if G !== nothing
        G[:] = ForwardDiff.gradient(th -> begin
            Vt = solve_ev(P_glob, th[1], th[2], th[3], δ, typeof(th[1]))
            σt = ccp(P_glob, Vt, th[1], th[2], th[3], δ)
            -sum(ck_glob .* log.(σt) .+ cr_glob .* log.(1 .- σt))
        end, θ)
    end
end

negll(θ)            = (f=Ref(zero(eltype(θ))); ll_and_grad!(f,nothing,θ); f[])
negll_grad!(G,θ)     = (f=Ref(zero(eltype(θ))); ll_and_grad!(f,G,θ); f[])

# ------------------------------------------------------
#   Main
# ------------------------------------------------------

function main(; n_buses=10_000_000, n_periods=100, seed=21, target=0.005) #21 for AEK Athens
    Random.seed!(seed)

    # simulation
    m = RustModel()
    set_parameters!(m, [2.4569, 0.03, 11.7257])
    update_transition_probs!(m, [0.0937,0.4475,0.4459,0.0127,0.0002])
    solve_value_function!(m)
    compute_choice_probabilities!(m)

    pid, bid, y, x = simulate_buses(m, n_buses, n_periods)
    @printf("Simulated %d buses × %d periods (%d obs)\n", n_buses, n_periods, length(pid))
    @printf("Total replacements: %d\n", sum(y))

    # sufficient stats
    global P_glob, ck_glob, cr_glob, dx_glob
    P_glob, ck_glob, cr_glob, dx_glob = build_transition_and_counts(bid, x, y, N)

    @printf("\nΔx|y=0 empirical pmf  —  first row of P\n")
    @printf("%4s  %8s  %10s\n","Δx","count","prob")
    for dx in 1:N
        count = dx_glob[dx]
        prob  = P_glob[1,dx]
        prob>0 && @printf("%+4d  %8d  %10.6f\n", dx-1, count, prob)
    end

    # MPEC
    θ1m, θ2m, RCm, llm = estimate_mpec(P_glob, ck_glob, cr_glob, δ)
    @printf("\nMPEC  θ₁=%.6f  θ₂=%.6f  RC=%.6f   LL=%.6f\n", θ1m, θ2m, RCm, llm)

    # NFXP - I use previous MPEC estimates as inital guesses; NFXP is basically a refinement
    θ0   = [θ1m, θ2m, RCm]
    opts = Optim.Options(iterations=1_000, show_trace=false, allow_f_increases=false)
    opt  = optimize(negll, negll_grad!, θ0, BFGS(), opts)
    θ̂    = Optim.minimizer(opt)
    @printf("NFXP  θ₁=%.6f  θ₂=%.6f  RC=%.6f   LL=%.6f\n", θ̂[1], θ̂[2], θ̂[3], -Optim.minimum(opt))

    H  = ForwardDiff.hessian(negll, θ̂)   # full-sample Hessian
    Σ  = inv(H)                           # covariance of (θ̂₁, θ̂₂, RĈ)
    se2_current = sqrt(Σ[2,2])            # current s.e. of θ̂₂;  θ̂₂ is the weakest link in the chain- it is not well identified
    @printf("θ₂ s.e. with %d buses: %.6f\n",
            n_buses, se2_current)

    # target s.e. that gives 90 % CI half-width of 0.01
    target_se  = target / 1.645              
    N_target   = ceil(Int, n_buses * (se2_current / target_se)^2)

    @printf("≈ %d buses give a 90%% CI of ±0.01 on θ₂.\n", N_target)

    return (θ̂ = θ̂,
            se2_current = se2_current,
            buses_needed = N_target)

end

# ------------------------------
# Execute 
# ------------------------------
main()
