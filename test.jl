using Plots

# 5-point stencil for the Laplacian with periodic boundary conditions
function laplacian(U)
    return circshift(U, (1, 0)) .+ circshift(U, (-1, 0)) .+
           circshift(U, (0, 1)) .+ circshift(U, (0, -1)) .- 4.0 .* U
end

function simulate_gray_scott()
    # Parameters from Milocco & Uller (2024)
    N = 50          # Grid size
    D1 = 0.32       # Diffusion rate for morphogen 1
    D2 = 0.06       # Diffusion rate for morphogen 2
    F = 0.032       # Production rate
    k = 0.060       # Degradation rate
    dt = 0.1        # Integration step (b = 0.1)
    steps = 100000    # Total steps for t = 500

    # Initial conditions: Background of x1=1, x2=0
    x1 = ones(N, N)
    x2 = zeros(N, N)

    # Introduce a perturbation in the center to kickstart the pattern
    center_size = 10
    start_idx = N ÷ 2 - center_size ÷ 2
    end_idx = N ÷ 2 + center_size ÷ 2
    
    # Adding a bit of noise helps break symmetry naturally
    x1[start_idx:end_idx, start_idx:end_idx] .= 0.50 .+ 0.1 .* rand(center_size+1, center_size+1)
    x2[start_idx:end_idx, start_idx:end_idx] .= 0.25 .+ 0.1 .* rand(center_size+1, center_size+1)

    # Main explicit Euler loop
    for step in 1:steps
        # Compute spatial derivatives
        lap_x1 = laplacian(x1)
        lap_x2 = laplacian(x2)

        # Non-linear reaction term
        reaction = x1 .* (x2 .^ 2)

        # Equation 6: Temporal derivatives
        dx1 = D1 .* lap_x1 .- reaction .+ F .* (1.0 .- x1)
        dx2 = D2 .* lap_x2 .+ reaction .- (F + k) .* x2

        # Update states
        x1 .= x1 .+ dt .* dx1
        x2 .= x2 .+ dt .* dx2
    end

    return x1, x2
end

# Run the simulation
x1_final, x2_final = simulate_gray_scott()

# Plot the concentration of morphogen 1 (Matches the paper's visual style)
heatmap(x1_final, c=:magma, title="Gray-Scott Morphogen 1", aspect_ratio=:equal, border=:none, legend=:none)