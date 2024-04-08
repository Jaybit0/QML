# ======== IMPORTS ========
# =========================

if !isdefined(Main, :GroverML)
    include("../Modules/GroverML.jl")
    using .GroverML
end

using Yao
using Yao.EasyBuild, YaoPlots

# ========== CODE ==========
# ==========================

# Initialize an empty circuit with 1 target lane and 4 model lanes
mcirc = chain(4, repeat(4, H, 2:4), control(2, 1=>Ry(π)), control(3, 1=>Ry(π/2)), control(4, 1=>Ry(π/4)))

target_lane = 1

# Apply Hadamard Gates on the lanes 2 -> 5
#hadamard(grover_circ, 2:5)

# Apply 4 controlled rotations with a granularity of pi/2 (max_rotation_rad / length(control_lanes))
#rotation(grover_circ, target_lane, control_lanes = 2:5; max_rotation_rad = 2*pi)

# We expect the first lane to return true
grover_circ = empty_circuit(1, 3)
yao_block(grover_circ, [1:4], mcirc)
out, main_circ, grov, oracle_function = auto_compute(grover_circ, [[true], [false], [true], [true], [true]], new_mapping_system = true)#, evaluate = false)

# Vizualize the main circuit
vizcircuit(main_circ)
# Uncomment this to vizualize the measured results
#measured = out |> r->measure(r; nshots=100000)
#plotmeasure(measured; oracle_function=oracle_function, sort=true, num_entries=14)