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

# Initialize an empty circuit with 2 target lane and 3 model lanes
mcirc = chain(6, repeat(6, H, 3:6), control(3, 1=>Ry(π)), control(4, 1=>Ry(π/2)), control(1, 2=>X), control((5, 1), 2=>Ry(π)),
                control((6, 1), 2=>Ry(π/2)))

model_lanes = 1:2
param_lanes = 3:6

grover = GroverMLBlock(mcirc, model_lanes, param_lanes, [[true, true], [true, false]]; log=true)

# Vizualize the main circuit
vizcircuit(grover.compiled_circuit.main_circuit)

# Uncomment this to vizualize the measured results
#plotmeasure(grover; sort=true, num_entries=14)