using FdeSolver
using Plots

# 1. Define the Fractional System Function
function gray_scott_frac(t, u, D1, D2, F_rate, k_rate, N)
    # FDEsolver passes 'u' as a flat vector; reshape to 2D grids
    N2 = N^2
    u1 = reshape(u[1:N2], N, N)
    u2 = reshape(u[N2+1:end], N, N)
    
    # 5-point Laplacian with periodic boundary conditions (torus)
    lap_u1 = circshift(u1, (1, 0)) .+ circshift(u1, (-1, 0)) .+
             circshift(u1, (0, 1)) .+ circshift(u1, (0, -1)) .- 4.0 .* u1
             
    lap_u2 = circshift(u2, (1, 0)) .+ circshift(u2, (-1, 0)) .+
             circshift(u2, (0, 1)) .+ circshift(u2, (0, -1)) .- 4.0 .* u2
    
    # Reaction kinetics
    reaction = u1 .* (u2 .^ 2)
    
    # Temporal derivatives
    du1 = D1 .* lap_u1 .- reaction .+ F_rate .* (1.0 .- u1)
    du2 = D2 .* lap_u2 .+ reaction .- (F_rate + k_rate) .* u2
    
    # FdeSolver expects a single flat vector returned
    return vcat(vec(du1), vec(du2))
end

# 2. Setup Parameters and Initial Conditions
N = 20 # Reduced grid size to manage history array footprint
D1, D2, F_rate, k_rate = 0.32, 0.06, 0.032, 0.060

u1_0 = ones(N, N)
u2_0 = zeros(N, N)

# Break symmetry in the center
c = 4 
start_idx = N ÷ 2 - c ÷ 2
end_idx = N ÷ 2 + c ÷ 2
u1_0[start_idx:end_idx, start_idx:end_idx] .= 0.50 .+ 0.1 .* rand(c+1, c+1)
u2_0[start_idx:end_idx, start_idx:end_idx] .= 0.25 .+ 0.1 .* rand(c+1, c+1)

y0 = vcat(vec(u1_0), vec(u2_0))

# 3. FdeSolver Settings
tSpan = [0.0, 500.0]              # Time horizon
# β = fill(0.8, length(y0))         # Fractional order (memory) of 0.8 for all states
# Morphogen 1 (u1) has strong memory (0.75)
β_u1 = fill(.9, N^2)

# Morphogen 2 (u2) has weak memory (0.95)
β_u2 = fill(.9, N^2)

# Combine them to match the structure of y0
β = vcat(β_u1, β_u2)

# 4. Execute the Predictor-Corrector Method
# Signature passes parameters directly to the function
t, Y = FDEsolver(gray_scott_frac, tSpan, y0, β, D1, D2, F_rate, k_rate, N, h = 0.1)

# 5. Extract and Plot the Final State
Y_final = Y[end, :]
u1_final = reshape(Y_final[1:N^2], N, N)

heatmap(u1_final, c=:magma, aspect_ratio=:equal, border=:none, legend=:none, title="Fractional Gray-Scott (α = 0.8)")