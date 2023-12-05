# ======== IMPORTS ========
# =========================

# Setup virtual environment as it doesn't work for me locally otherwise
# These three lines of code must be executed before anything else, even imports
include("../Modules/SetupTool.jl")

using .SetupTool

setupPackages(false, update_registry = false)

using Revise

Revise.includet("../Modules/GroverML.jl")

using .GroverML

Revise.includet("../Modules/GroverCircuitBuilder.jl")

using .GroverCircuitBuilder

Revise.includet("../Modules/GroverPlotting.jl")

using .GroverPlotting

configureYaoPlots()

using Yao
using Yao.EasyBuild, YaoPlots

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
out, main_circ, grov = auto_compute(grover_circ, criterion, forced_grover_iterations = 1)#, evaluate = false)

# Vizualize the main circuit
vizcircuit(grov)

# Uncomment this to vizualize the measured results
#measured = out |> r->measure(r; nshots=100000)
#plotmeasure(measured)