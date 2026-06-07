using Plots

# 5-point stencil for the Laplacian with periodic boundary conditions
function laplacian(U)
    return circshift(U, (1, 0)) .+ circshift(U, (-1, 0)) .+
           circshift(U, (0, 1)) .+ circshift(U, (0, -1)) .- 4.0 .* U
end

function animate_gray_scott()
    # Parameters from Milocco & Uller (2024)
    N = 100          
    D1 = 0.32       
    D2 = 0.06       
    F = 0.032       
    k = 0.060       
    dt = 0.1        
    steps = 50000    
    save_interval = 50 # Capture a frame every 50 steps

    x1 = ones(N, N)
    x2 = zeros(N, N)

    # Initial perturbation
    center_size = 10
    start_idx = N ÷ 2 - center_size ÷ 2
    end_idx = N ÷ 2 + center_size ÷ 2
    x1[start_idx:end_idx, start_idx:end_idx] .= 0.50 .+ 0.1 .* rand(center_size+1, center_size+1)
    x2[start_idx:end_idx, start_idx:end_idx] .= 0.25 .+ 0.1 .* rand(center_size+1, center_size+1)

    # Initialize the animation object
    anim = Animation()

    for step in 1:steps
        lap_x1 = laplacian(x1)
        lap_x2 = laplacian(x2)

        reaction = x1 .* (x2 .^ 2)

        dx1 = D1 .* lap_x1 .- reaction .+ F .* (1.0 .- x1)
        dx2 = D2 .* lap_x2 .+ reaction .- (F + k) .* x2

        x1 .= x1 .+ dt .* dx1
        x2 .= x2 .+ dt .* dx2

        # Save the current state as a frame
        if step % save_interval == 0
            plt = heatmap(x1, c=:magma, aspect_ratio=:equal, border=:none, 
                          legend=:none, title="Time step: $step", clims=(0, 1))
            frame(anim, plt)
        end
    end

    # Export the animation
    gif(anim, "gray_scott_evolution.gif", fps=15)
    println("Evolution saved to 'gray_scott_evolution.gif'")
end

animate_gray_scott()