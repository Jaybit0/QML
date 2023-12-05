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

# Initialize an empty circuit with 2 target lanes and 4 model lanes
grover_circ = empty_circuit(2, 3)

hadamard(grover_circ, model_lanes(grover_circ), false)
learned_rotation(grover_circ, target_lanes(grover_circ)[1], model_lanes(grover_circ)[1:2], true)
not(grover_circ, 2, true; control_lanes = [model_lanes(grover_circ)[2:3]])

#main_circ = compile_circuit(grover_circ, inv = false)
out, main_circ, grov = auto_compute(grover_circ, [[(false, nothing), (false, true)], [(true, nothing), (true, false)]])

# Visualize the main circuit
vizcircuit(main_circ)

#measured = out |> r->measure(r; nshots=1000)
#plotmeasure(measured)