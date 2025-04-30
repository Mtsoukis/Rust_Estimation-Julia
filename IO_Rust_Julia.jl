using Pkg
import Pkg
using JuMP
using CairoMakie
using BenchmarkTools
using CSV
using DataFrames
using DataFramesMeta
using Chain
using StatsBase
import StatsBase: countmap
import Printf: @printf
using Printf
using PkgVersion
using Ipopt
using Optim
using ShiftedArrays
import ShiftedArrays: lag
using LinearAlgebra
using Observables
using BenchmarkTools
using GLMakie

const DATAFILE = "/Users/mariostsoukis/Documents/NYU/IO_II/HW_3/rust_data_2020.csv"
const N = 201
const DF = CSV.File(DATAFILE; types=Dict(:x_it=>Int16, :y_it=>UInt8)) |> DataFrame

function build_transition_and_counts(df::DataFrame, N::Int)
    x = Int.(df.x_it)
    y = Int.(df.y_it)
    P_int = zeros(Int, N, N)
    counts_keep = zeros(Int, N)
    counts_rep = zeros(Int, N)
    @inbounds @simd for i in 2:length(x)
        if df.bus_id[i] == df.bus_id[i-1]
            xp = x[i-1] + 1
            xn = x[i] + 1
            if y[i] == 0
                P_int[xp, xn] += 1
                counts_keep[xp] += 1
            else
                counts_rep[xp] += 1
            end
        end
    end
    P = Matrix{Float64}(undef, N, N)
    @inbounds for r in 1:N
        s = sum(P_int[r, :])
        if s == 0
            P[r, :] .= 0.0
            P[r, r] = 1.0
        else
            P[r, :] .= P_int[r, :] ./ s
        end
    end
    return P, counts_keep, counts_rep
end

const P_, counts_keep_, counts_rep_ = build_transition_and_counts(DF, N)

function compute_pmf(df::DataFrame, y_val::Int)
    df2 = @chain df begin
        groupby(:bus_id)
        @transform(_, :delta_x = vcat(missing, Base.diff(:x_it)))
    end
    df3 = filter(r -> r.y_it == y_val && !ismissing(r.delta_x), df2)
    cnts = countmap(df3.delta_x)
    xs = sort(collect(keys(cnts)))
    tot = sum(values(cnts))
    DataFrame(delta_x = xs,
              count = [cnts[x] for x in xs],
              prob = [cnts[x] / tot for x in xs])
end

function estimate_mpec(P, counts_keep, counts_rep, δ)
    N = size(P, 1)
    model = Model(Ipopt.Optimizer)
    set_silent(model)
    @variable(model, θ₁ ≥ 1e-6)
    @variable(model, θ₂ ≥ 1e-6)
    @variable(model, β ≥ 1e-6)
    @variable(model, V[1:N])
    @variable(model, v_keep[1:N])
    @variable(model, v_rep)
    @NLexpression(model, x_grid[j=1:N], (j-1)/100)
    @NLexpression(model, EV_keep[j=1:N], sum(P[j,k] * V[k] for k in 1:N))
    @NLconstraint(model, [j=1:N],
        v_keep[j] + θ₁*x_grid[j] + θ₂*x_grid[j]^2 - δ * EV_keep[j] == 0)
    @NLconstraint(model, v_rep + β - δ * V[1] == 0)
    @NLexpression(model, denom[j=1:N], exp(v_keep[j]) + exp(v_rep))
    @NLexpression(model, σ_keep[j=1:N], exp(v_keep[j]) / denom[j])
    @NLconstraint(model, [j=1:N], V[j] == log(denom[j]))
    @NLobjective(model, Min, -sum(
        counts_keep[i] * log(σ_keep[i]) +
        counts_rep[i]  * log(1 - σ_keep[i]) for i in 1:N))
    set_start_value.( (θ₁, θ₂, β), 1.0 )
    set_start_value.(V, -5.0)
    set_start_value.(v_keep, -5.0)
    set_start_value(v_rep, -15.0)
    set_optimizer_attribute(model, "tol", 1e-6)
    set_optimizer_attribute(model, "max_iter", 1500)
    optimize!(model)
    θ1_m = value(θ₁)
    θ2_m = value(θ₂)
    β_m  = value(β)
    ll   = -objective_value(model)
    return θ1_m, θ2_m, β_m, ll
end

function solve_ev_nfxp(P, θ₁, θ₂, β, δ; tol=1e-8)
    N = size(P,1)
    x = range(0, N-1; step=1) ./ 100
    V = zeros(Float64, N)
    diff = Inf
    while diff > tol
        EV = P * V
        v_k = -(θ₁ .* x .+ θ₂ .* x.^2) .+ δ .* EV
        v_r = -β .+ δ .* V[1]
        V_new = log.(exp.(v_k) .+ exp.(v_r))
        diff = maximum(abs.(V_new .- V))
        V .= V_new
    end
    return V
end

function compute_ccp_nfxp(V, θ₁, θ₂, β, P, δ)
    N = length(V)
    x = range(0, N-1; step=1) ./ 100
    EV = P * V
    v_k = -(θ₁ .* x .+ θ₂ .* x.^2) .+ δ .* EV
    v_r = -β .+ δ .* V[1]
    return exp.(v_k) ./ (exp.(v_k) .+ exp.(v_r))
end

function neg_loglik_nfxp(p, P, c_keep, c_rep, δ)
    θ₁, θ₂, β = p
    (θ₁ ≤ 0 || θ₂ ≤ 0 || β ≤ 0) && return Inf
    V = solve_ev_nfxp(P, θ₁, θ₂, β, δ)
    σ = compute_ccp_nfxp(V, θ₁, θ₂, β, P, δ)
    -sum(c_keep .* log.(σ) .+ c_rep .* log.(1 .- σ))
end

function main_part_rust(P, counts_keep, counts_rep; δ::Float64)
    θ1_m, θ2_m, β_m, ll_mpec = estimate_mpec(P, counts_keep, counts_rep, δ)
    @printf("MPEC Results:\n θ₁ = %.6f, θ₂ = %.6f, β = %.6f, LL = %.6f\n",
            θ1_m, θ2_m, β_m, ll_mpec)
    res = optimize(
        x -> neg_loglik_nfxp(x, P, counts_keep, counts_rep, δ),
        [θ1_m, θ2_m, β_m],
        NelderMead(),
        Optim.Options(show_trace = false)
    )
    ll_nfxp = -res.minimum
    θ1_hat, θ2_hat, β_hat = res.minimizer
    @printf("NFXP Results:\n θ₁ = %.6f, θ₂ = %.6f, β = %.6f, LL = %.6f\n",
            θ1_hat, θ2_hat, β_hat, ll_nfxp)
    return θ1_m, θ2_m, β_m, θ1_hat, θ2_hat, β_hat
end

θ1_m, θ2_m, β_m, θ1_hat, θ2_hat, β_hat = main_part_rust(P_, counts_keep_, counts_rep_; δ=0.975)

chris_θ = (2.4569, 0.03, 11.7257)
my_θ = (θ1_hat, θ2_hat, β_hat)
nll_chris = -neg_loglik_nfxp(chris_θ, P_, counts_keep_, counts_rep_, 0.975)
my_ll = -861.875603
@printf("My θ:       θ₁ = %.4f, θ₂ = %.4f, β = %.4f\n", my_θ...)
@printf("Chris θ:    θ₁ = %.4f, θ₂ = %.4f, β = %.4f\n", chris_θ...)
@printf("My -LL:     %.6f\n", my_ll)
@printf("Chris -LL:  %.6f\n", nll_chris)

function plot_ev_keep!(
    P,
    θ1_m, θ2_m, β_m,
    θ1_hat, θ2_hat, β_hat,
    δ;
    #savepath="EV_keep_MPEC_vs_NFXP.svg",
    backend="svg",
    dpi=800
)
    V_mpec = solve_ev_nfxp(P, θ1_m, θ2_m, β_m, δ)
    V_nfxp = solve_ev_nfxp(P, θ1_hat, θ2_hat, β_hat, δ)
    EV_mpec = P * V_mpec
    EV_nfxp = P * V_nfxp
    states = 0:(size(P,1)-1)
    CairoMakie.activate!(; type=backend)
    fig = Figure(size=(800,600), dpi=dpi)
    ax1 = Axis(fig[1,1]; xlabel="State j", ylabel="E[V′ | y = 0]", title="Expected Value of not Replacing Engine")
    lines!(ax1, states, EV_mpec; label="MPEC", linewidth=2)
    lines!(ax1, states, EV_nfxp; label="NFXP", linewidth=2, linestyle=:dash)
    axislegend(ax1, position=:rb)
    ax2 = Axis(fig[2,1]; xlabel="State j", ylabel="MPEC − NFXP")
    lines!(ax2, states, EV_mpec .- EV_nfxp; linewidth=2)
    hlines!(ax2, [0]; linestyle=:dot, color=:gray)
    #save(savepath, fig)
    #println("Figure saved to $(savepath)")
    return fig
end

function run_and_plot(; N=201, δ=0.975, backend="png", dpi=800)
    fig = plot_ev_keep!(
        P_,
        θ1_m, θ2_m, β_m,
        θ1_hat, θ2_hat, β_hat,
        δ;
        backend=backend,
        dpi=dpi,
    )
    display(fig)
    return nothing
end

run_and_plot()

function plot_replacement_demand(
    β_range = range(2.0, stop=20.0, length=200);
    δ=0.975,
    #savepath="replacement_demand.png",
    backend="png",
    dpi=600
)
    θ1_m, θ2_m, β_m, _ = estimate_mpec(P_, counts_keep_, counts_rep_, δ)
    totals = counts_keep_ .+ counts_rep_
    π_emp  = totals ./ sum(totals)
    demands = Float64[]
    for β in β_range
        V = solve_ev_nfxp(P_, θ1_m, θ2_m, β, δ)
        rep = 1 .- compute_ccp_nfxp(V, θ1_m, θ2_m, β, P_, δ)
        push!(demands, dot(rep, π_emp))
    end
    CairoMakie.activate!(; type=backend)
    fig = Figure(size=(800,600), dpi=dpi)
    ax = Axis(fig[1,1]; xlabel="Price β", ylabel="Replacement Demand", title="Demand curve (Estimated Parameters)")
    lines!(ax, β_range, demands; label="δ = $(round(δ, sigdigits=3))", linewidth=2)
    hlines!(ax, [dot(1 .- compute_ccp_nfxp(solve_ev_nfxp(P_, θ1_m, θ2_m, β_m, δ), θ1_m, θ2_m, β_m, P_, δ), π_emp)]; linestyle=:dash, label="at estimated β")
    axislegend(ax)
    ##save(savepath, fig)
    #println("Figure saved to $(savepath)")
    return fig
end

plot_replacement_demand()

function plot_replacement_demand_multi(
    δs = [0.975, 0.01];
    β_range = range(1.0, stop=12.0, length=200),
    #savepath = "replacement_demand_comparison.png",
    backend = "png",
    dpi = 300
)
    π_emp = (counts_keep_ .+ counts_rep_) ./ sum(counts_keep_ .+ counts_rep_)
    demand_profiles = Dict{Float64, Vector{Float64}}()
    for δ in δs
        θ1_m, θ2_m, β_m, _ = estimate_mpec(P_, counts_keep_, counts_rep_, δ)
        demands = Float64[]
        for β in β_range
            V = solve_ev_nfxp(P_, θ1_m, θ2_m, β, δ)
            rep = 1 .- compute_ccp_nfxp(V, θ1_m, θ2_m, β, P_, δ)
            push!(demands, dot(rep, π_emp))
        end
        demand_profiles[δ] = demands
    end
    CairoMakie.activate!(; type=backend)
    fig = Figure(size=(800,600), dpi=dpi)
    ax = Axis(fig[1,1]; xlabel="Price β", ylabel="Replacement Demand", title="Demand Curves")
    for δ in δs
        lines!(ax, β_range, demand_profiles[δ]; label="δ = $(δ)", linewidth=2)
    end
    axislegend(ax, position=:rt)
    #save(savepath, fig)
    #println("Figure saved to $(savepath)")
    return fig
end

plot_replacement_demand_multi()

function animate_replacement_demand_lift(
    δ_values = range(0.01, stop=0.99, length=60);
    β_range = range(2.0, stop=20.0, length=200),
    #savepath = "replacement_demand_lift.gif",
    fps = 12
)
    π_emp = (counts_keep_ .+ counts_rep_) ./ sum(counts_keep_ .+ counts_rep_)
    δ_obs = Observable(first(δ_values))
    demand_obs = @lift begin
        δ = $δ_obs
        θ1, θ2, βm, _ = estimate_mpec(P_, counts_keep_, counts_rep_, δ)
        [ dot(1 .- compute_ccp_nfxp(solve_ev_nfxp(P_, θ1, θ2, β, δ), θ1, θ2, β, P_, δ), π_emp) for β in β_range ]
    end
    marker_obs = @lift begin
        δ = $δ_obs
        θ1, θ2, βm, _ = estimate_mpec(P_, counts_keep_, counts_rep_, δ)
        dot(1 .- compute_ccp_nfxp(solve_ev_nfxp(P_, θ1, θ2, βm, δ), θ1, θ2, βm, P_, δ), π_emp)
    end
    fig = Figure(size=(800,600))
    ax = Axis(fig[1,1]; xlabel="Price β", ylabel="Replacement Demand", title="δ = ?")
    lines!(ax, β_range, demand_obs; color=:blue, label="Demand")
    hlines!(ax, marker_obs; linestyle=:dash, label="at observed β")
    axislegend(ax, position=:rt)
    #record(fig, savepath, δ_values; framerate=fps) do δ
        δ_obs[] = δ
        ax.title = "Demand curve as δ evolves (δ = $(round(δ, sigdigits=3)))"
    end
    #println("Saved animation to $(savepath)")
    return nothing
