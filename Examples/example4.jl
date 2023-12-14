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

distribution = [true, true, false, false]

# Initialize an empty circuit with 2 target lanes and 4 model lanes
grover_circ = empty_circuit(1, 4)

# Apply Hadamard Gates on the lanes 3 -> 6
hadamard(grover_circ, model_lanes(grover_circ))

# Apply 3 controlled rotations on the first lane with a granularity of pi/4 (max_rotation_rad / 2^length(control_lanes))
block, meta = learned_rotation(grover_circ, target_lanes(grover_circ)[1], model_lanes(grover_circ))
#meta.data["lane"] = 1
#meta.data["batch"] = 1
#meta.manipulator = (block, meta, inv) -> distribution[meta.data["batch"]]

# We expect the first lane to return true and the second lane to return false
# As we use multiple target lanes, auto_compute automatically inserts a lane below the target lanes which encode the criterions to this lane
# The reflection is done with respect to the inserted lane
# As we have provided a wrong inverse, the process should fail and auto_compute should automatically identify the wrong inverse
out, main_circ, grov = auto_compute(grover_circ, [[true], [true], [false], [false]])

# Visualize the main circuit
vizcircuit(main_circ)
#vizcircuit(chain(6, put(1:6 => compile_block(grover_circ, yao, meta)), put(1:6 => compile_block(grover_circ, yao, meta; inv=true))))

# Uncomment this to vizualize the measured results
#measured = out |> r->measure(r; nshots=100000)
#plotmeasure(measured)