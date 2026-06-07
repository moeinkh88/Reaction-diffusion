using FdeSolver
using Plots

# 1. Define the System (Gray-Scott)
function gray_scott_sys(t, u, D1, D2, F_rate, k_rate, N)
    N2 = N^2
    u1 = reshape(u[1:N2], N, N)
    u2 = reshape(u[N2+1:end], N, N)
    
    # 5-point Laplacian with periodic boundaries
    lap_u1 = circshift(u1, (1, 0)) .+ circshift(u1, (-1, 0)) .+
             circshift(u1, (0, 1)) .+ circshift(u1, (0, -1)) .- 4.0 .* u1
             
    lap_u2 = circshift(u2, (1, 0)) .+ circshift(u2, (-1, 0)) .+
             circshift(u2, (0, 1)) .+ circshift(u2, (0, -1)) .- 4.0 .* u2
    
    reaction = u1 .* (u2 .^ 2)
    
    du1 = D1 .* lap_u1 .- reaction .+ F_rate .* (1.0 .- u1)
    du2 = D2 .* lap_u2 .+ reaction .- (F_rate + k_rate) .* u2
    
    return vcat(vec(du1), vec(du2))
end

# 2. Setup Parameters & Identical Initial Conditions
N = 30 # Grid size (Keep at 20-30 for fast FDE testing, scale up later)
D1, D2, F_rate, k_rate = 0.32, 0.06, 0.032, 0.060
tSpan = [0.0, 500.0]
dt = 0.1 # Slightly larger step to speed up FDE solver for the GIF

u1_0 = ones(N, N)
u2_0 = zeros(N, N)

# Break symmetry in the center
c = 6 
start_idx = N ÷ 2 - c ÷ 2
end_idx = N ÷ 2 + c ÷ 2
u1_0[start_idx:end_idx, start_idx:end_idx] .= 0.50 .+ 0.1 .* rand(c+1, c+1)
u2_0[start_idx:end_idx, start_idx:end_idx] .= 0.25 .+ 0.1 .* rand(c+1, c+1)

y0 = vcat(vec(u1_0), vec(u2_0))

# 3. Define Memory Dimensions
# No Memory (Markovian, integer derivative)
β_nomem = fill(1.0, length(y0)) 

# With Memory (Path-dependent, fractional derivative)
# β_mem = fill(0.9, length(y0))   
# Morphogen 1 (u1) has strong memory (0.75)
β_u1 = fill(.8, N^2)

# Morphogen 2 (u2) has weak memory (0.95)
β_u2 = fill(1.0, N^2)

# Combine them to match the structure of y0
β_mem = vcat(β_u1, β_u2)

# 4. Run Both Simulations
println("Solving Memoryless System (β = 1.0)...")
t_nomem, Y_nomem = FDEsolver(gray_scott_sys, tSpan, y0, β_nomem, D1, D2, F_rate, k_rate, N, h=dt)

println("Solving System with Memory (β = 0.8)...")
t_mem, Y_mem = FDEsolver(gray_scott_sys, tSpan, y0, β_mem, D1, D2, F_rate, k_rate, N, h=dt)

# ---------------------------------------------------------
# VISUALIZATION 1: Static 2x2 Figure (First vs. Last Moment)
# ---------------------------------------------------------
println("Generating static comparison figure...")

# Extract states
u1_start = reshape(y0[1:N^2], N, N)
u1_end_nomem = reshape(Y_nomem[end, 1:N^2], N, N)
u1_end_mem = reshape(Y_mem[end, 1:N^2], N, N)

# Common heatmap settings to keep code clean
hm_args = (c=:magma, aspect_ratio=:equal, border=:none, legend=:none, clims=(0,1))

p_start_nomem = heatmap(u1_start; title="No Memory (t=0)", hm_args...)
p_end_nomem   = heatmap(u1_end_nomem; title="No Memory (t=500)", hm_args...)
p_start_mem   = heatmap(u1_start; title="With Memory (t=0)", hm_args...)
p_end_mem     = heatmap(u1_end_mem; title="With Memory (t=500)", hm_args...)

static_plot = plot(p_start_nomem, p_end_nomem, p_start_mem, p_end_mem, 
                   layout=(2,2), size=(800, 800), margin=5Plots.mm)

savefig(static_plot, "Memory_Comparison_Static.png")

# ---------------------------------------------------------
# VISUALIZATION 2: Side-by-Side Evolution GIF
# ---------------------------------------------------------
println("Generating side-by-side animation...")

anim = Animation()
num_steps = length(t_nomem)
save_interval = max(1, num_steps ÷ 100) # Aim for ~100 frames

for i in 1:save_interval:num_steps
    # Extract current frames
    frame_nomem = reshape(Y_nomem[i, 1:N^2], N, N)
    frame_mem   = reshape(Y_mem[i, 1:N^2], N, N)
    
    # Plot side-by-side
    p1 = heatmap(frame_nomem; title="No Memory (β=1.0)", hm_args...)
    p2 = heatmap(frame_mem; title="With Memory (β=0.8)", hm_args...)
    
    time_stamp = round(t_nomem[i], digits=1)
    combined = plot(p1, p2, layout=(1,2), size=(800, 400), 
                    plot_title="Evolution at t = $time_stamp")
    
    frame(anim, combined)
end

# Save GIF
gif(anim, "Memory_Comparison_Evolution.gif", fps=15)
println("Done! Files saved as Memory_Comparison_Static.png and Memory_Comparison_Evolution.gif")