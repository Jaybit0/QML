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
grover_circ = empty_circuit(2, 4)

custom_block = chain(2, put(1 => Rz(pi)), put(2 => Rz(pi)))
custom_block_inv = chain(2, put(2 => Rz(-pi)), put(1 => Rz(-pi)))
yao_block(grover_circ, [1:2, 1:2], custom_block, custom_block_inv, control_lanes=[3:4, 5:6])

# Visualize the main circuit
vizcircuit(compile_circuit(grover_circ))

#measured = out |> r->measure(r; nshots=1000)
#plotmeasure(measured)