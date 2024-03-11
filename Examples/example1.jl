# ======== IMPORTS ========
# =========================

include("../Modules/GroverML.jl")
include("../Modules/GroverCircuitBuilder.jl")
include("../Modules/GroverPlotting.jl")

using Yao
using Yao.EasyBuild, YaoPlots
using .GroverML
using .GroverCircuitBuilder
using .GroverPlotting

# ========== CODE ==========
# ==========================

# Initialize an empty circuit with 1 target lane and 4 model lanes
grover_circ = empty_circuit(1, 4)

target_lane = 1

# Apply Hadamard Gates on the lanes 2 -> 5
hadamard(grover_circ, 2:5)

# Apply 4 controlled rotations with a granularity of pi/2 (max_rotation_rad / length(control_lanes))
rotation(grover_circ, target_lane, control_lanes = 2:5; max_rotation_rad = 2*pi)

# We expect the first lane to return true
criterion = [true]
out, main_circ, grov, oracle_function = auto_compute(grover_circ, criterion)#, evaluate = false)

# Vizualize the main circuit
#vizcircuit(grov)
# Uncomment this to vizualize the measured results
measured = out |> r->measure(r; nshots=100000)
plotmeasure(measured; oracle_function=oracle_function, sort=true, num_entries=14)