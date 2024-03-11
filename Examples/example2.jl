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

# Initialize an empty circuit with 2 target lanes and 4 model lanes
grover_circ = empty_circuit(2, 4)

# Apply Hadamard Gates on the lanes 3 -> 6
hadamard(grover_circ, 3:6)

# Apply 3 controlled rotations on the first lane with a granularity of pi/4 (max_rotation_rad / 2^length(control_lanes))
block, meta = learned_rotation(grover_circ, 1, 3:5)
meta.data["batch"] = 1
meta.data["lane"] = 1
meta.manipulator(block, meta, inv) = meta.data["batch"] == 1
# Apply 1 controlled rotation on the second lane with a granularity of pi (max_rotation_rad / 2^length(control_lanes))
learned_rotation(grover_circ, 2, 6)

# Apply a controlled negation to the second lane
not(grover_circ, 2; control_lanes = 1)

# We expect the first lane to return true and the second lane to return false
# As we use multiple target lanes, auto_compute automatically inserts a lane below the target lanes which encode the criterions to this lane
# The reflection is done with respect to the inserted lane
criterion = [true, true]
out, main_circ, grov, oracle_function = auto_compute(grover_circ, criterion, evaluate = true)

# Visualize the main circuit
vizcircuit(main_circ)

# Uncomment this to vizualize the measured results
#measured = out |> r->measure(r; nshots=100000)
#plotmeasure(measured; oracle_function=oracle_function, sort=true, num_entries=10)