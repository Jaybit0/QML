# ======== IMPORTS ========
# =========================

if !isdefined(Main, :QML)
    include("../Modules/QML.jl")
    using .QML
end

using Yao
using Yao.EasyBuild, YaoPlots

# ========== CODE ==========
# ==========================

model_lanes = 1
param_lanes = 2:4

bernoulli_measurements = [[false], [false], [true], [false]]

# Initialize an empty circuit with 1 model lane and 3 parameter lanes
mcirc = chain(4, repeat(H, param_lanes), control(2, 1=>Ry(π)), control(3, 1=>Ry(π/2)),control(4, 1=>Ry(π/4)))

grover = QMLBlock(mcirc, model_lanes, param_lanes, bernoulli_measurements; log=true)

# Vizualize the main circuit (uncomment to see the circuit)
#vizcircuit(grover.compiled_circuit.main_circuit)

# Uncomment this to vizualize the measured results
plotmeasure(grover; sort=true, num_entries=8)