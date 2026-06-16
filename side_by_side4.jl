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
N = 50 # Grid size 
D1, D2, F_rate, k_rate = 0.32, 0.06, 0.032, 0.060
tSpan = [0.0, 3000.0]
dt = 0.1 

u1_0 = ones(N, N)
u2_0 = zeros(N, N)

# Break symmetry in the center
c = 10 
start_idx = N ÷ 2 - c ÷ 2
end_idx = N ÷ 2 + c ÷ 2
u1_0[start_idx:end_idx, start_idx:end_idx] .= 0.50 .+ 0.1 .* rand(c+1, c+1)
u2_0[start_idx:end_idx, start_idx:end_idx] .= 0.25 .+ 0.1 .* rand(c+1, c+1)

y0 = vcat(vec(u1_0), vec(u2_0))

# 3. Define Memory Dimensions
# Row 1: No Memory (Markovian, integer derivative)
β_nomem = fill(1.0, length(y0)) 

# Row 2: Identical Memory (Richness = 1)
β_u1_r1 = fill(0.8, N^2)
β_u2_r1 = fill(0.8, N^2)
β_mem_r1 = vcat(β_u1_r1, β_u2_r1)

# Row 3: Mismatched Memory (Richness = 2)
β_u1_r2 = fill(0.8, N^2)
β_u2_r2 = fill(0.95, N^2)
β_mem_r2 = vcat(β_u1_r2, β_u2_r2)

# 4. Run All Three Simulations
println("Solving Memoryless System (β = 1.0, 1.0)...")
t_nomem, Y_nomem = FDEsolver(gray_scott_sys, tSpan, y0, β_nomem, D1, D2, F_rate, k_rate, N, h=dt)

println("Solving Identical Memory System (β = 0.8, 0.8)...")
t_mem_r1, Y_mem_r1 = FDEsolver(gray_scott_sys, tSpan, y0, β_mem_r1, D1, D2, F_rate, k_rate, N, h=dt)

println("Solving Mismatched Memory System (β = 0.8, 0.95)...")
t_mem_r2, Y_mem_r2 = FDEsolver(gray_scott_sys, tSpan, y0, β_mem_r2, D1, D2, F_rate, k_rate, N, h=dt)

# ---------------------------------------------------------
# VISUALIZATION 1: Static 3x3 Figure (t=300, t=600, t=1000)
# ---------------------------------------------------------
println("Generating static comparison figure...")

# Find the array indices that correspond to t=300 and t=600
idx_300 = findfirst(>=(300.0), t_nomem)
idx_600 = findfirst(>=(600.0), t_nomem)

# Extract states for No Memory
u1_300_nomem = reshape(Y_nomem[idx_300, 1:N^2], N, N)
u1_600_nomem = reshape(Y_nomem[idx_600, 1:N^2], N, N)
u1_1000_nomem = reshape(Y_nomem[end, 1:N^2], N, N)

# Extract states for Identical Memory (Richness 1)
u1_300_r1 = reshape(Y_mem_r1[idx_300, 1:N^2], N, N)
u1_600_r1 = reshape(Y_mem_r1[idx_600, 1:N^2], N, N)
u1_1000_r1 = reshape(Y_mem_r1[end, 1:N^2], N, N)

# Extract states for Mismatched Memory (Richness 2)
u1_300_r2 = reshape(Y_mem_r2[idx_300, 1:N^2], N, N)
u1_600_r2 = reshape(Y_mem_r2[idx_600, 1:N^2], N, N)
u1_1000_r2 = reshape(Y_mem_r2[end, 1:N^2], N, N)

# Common heatmap settings
hm_args = (c=:viridis, aspect_ratio=:equal, border=:none, legend=:none, clims=(0,1))

# Top Row: No Memory
p_300_nomem = heatmap(u1_300_nomem; title="No Memory (t=300)", hm_args...)
p_600_nomem = heatmap(u1_600_nomem; title="No Memory (t=600)", hm_args...)
p_1000_nomem = heatmap(u1_1000_nomem; title="No Memory (t=3000)", hm_args...)

# Middle Row: Identical Memory
p_300_r1 = heatmap(u1_300_r1; title="Identical Memory (t=300)", hm_args...)
p_600_r1 = heatmap(u1_600_r1; title="Identical Memory (t=600)", hm_args...)
p_1000_r1 = heatmap(u1_1000_r1; title="Identical Memory (t=3000)", hm_args...)

# Bottom Row: Mismatched Memory
p_300_r2 = heatmap(u1_300_r2; title="Mismatched Memory (t=300)", hm_args...)
p_600_r2 = heatmap(u1_600_r2; title="Mismatched Memory (t=600)", hm_args...)
p_1000_r2 = heatmap(u1_1000_r2; title="Mismatched Memory (t=3000)", hm_args...)

# Combine into a 3x3 grid
static_plot = plot(p_300_nomem, p_600_nomem, p_1000_nomem, 
                   p_300_r1, p_600_r1, p_1000_r1, 
                   p_300_r2, p_600_r2, p_1000_r2, 
                   layout=(3,3), size=(1200, 1200), margin=5Plots.mm)

savefig(static_plot, "Memory_Comparison_Static5.png")
savefig(static_plot, "Memory_Comparison_Static5.svg")

# ---------------------------------------------------------
# VISUALIZATION 2: Side-by-Side Evolution GIF
# ---------------------------------------------------------
println("Generating side-by-side animation...")

anim = Animation()
num_steps = length(t_nomem)
save_interval = max(1, num_steps ÷ 200) # Aim for ~100 frames

for i in 1:save_interval:num_steps
    # Extract current frames
    frame_nomem = reshape(Y_nomem[i, 1:N^2], N, N)
    frame_r1    = reshape(Y_mem_r1[i, 1:N^2], N, N)
    frame_r2    = reshape(Y_mem_r2[i, 1:N^2], N, N)
    
    # Plot side-by-side
    p1 = heatmap(frame_nomem; title="No Memory (β=1.0, 1.0)", hm_args...)
    p2 = heatmap(frame_r1; title="Identical Memory (β=0.8, 0.8)", hm_args...)
    p3 = heatmap(frame_r2; title="Mismatched Memory (β=0.8, 0.95)", hm_args...)
    
    time_stamp = round(t_nomem[i], digits=1)
    # Increased width to 1200 to accommodate the third panel
    combined = plot(p1, p2, p3, layout=(1,3), size=(1200, 400), 
                    plot_title="Evolution at t = $time_stamp")
    
    frame(anim, combined)
end

# Save GIF
gif(anim, "Memory_Comparison_Evolution5.gif", fps=5)
println("Done! Files saved as Memory_Comparison_Static5.png/svg and Memory_Comparison_Evolution5.gif")