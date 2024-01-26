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

# Apply Hadamard Gates on the lanes 3 -> 6
hadamard(grover_circ, 3:6)

# We can now pass custom blocks to our circuit
# Note that we provide an faulty inverse block on purpose
custom_block = chain(2, put(1 => Rz(pi)), put(2 => Rz(pi)))
custom_block_inv = chain(2, put(1 => Rz(-pi/2)), put(2 => Rz(-pi/2)))
yao, meta = yao_block(grover_circ, [[1, 3], 2:3, 3:4], custom_block, custom_block_inv, control_lanes = [[2, 4], 4:5, 5:6])

# Apply 3 controlled rotations on the first lane with a granularity of pi/4 (max_rotation_rad / 2^length(control_lanes))
learned_rotation(grover_circ, 1, 3:5)
# Apply 1 controlled rotation on the second lane with a granularity of pi (max_rotation_rad / 2^length(control_lanes))
learned_rotation(grover_circ, 2, 6)

# Apply a controlled negation to the second lane
not(grover_circ, 2; control_lanes = 1)

# We expect the first lane to return true and the second lane to return false
# As we use multiple target lanes, auto_compute automatically inserts a lane below the target lanes which encode the criterions to this lane
# The reflection is done with respect to the inserted lane
# As we have provided a wrong inverse, the process should fail and auto_compute should automatically identify the wrong inverse
criterion = [true, true]
out, main_circ, grov, oracle_function = auto_compute(grover_circ, criterion, evaluate = true)

# Visualize the main circuit
vizcircuit(main_circ)
#vizcircuit(chain(6, put(1:6 => compile_block(grover_circ, yao, meta)), put(1:6 => compile_block(grover_circ, yao, meta; inv=true))))

# Uncomment this to vizualize the measured results
#measured = out |> r->measure(r; nshots=100000)
#plotmeasure(measured; oracle_function=oracle_function)