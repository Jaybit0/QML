# ======== IMPORTS ========
# =========================
include("../Modules/OAACircuitBuilder.jl")

using Yao
using Yao.EasyBuild, YaoPlots

# num_model_lanes = 2
rotation_precision = 2

training_data = [[0,0],[1,0], [1,1]]
# training_data = [[0,0, 1],[1,0, 0], [1,1, 0]]

model = create_oaa_circuit(training_data, rotation_precision)

vizcircuit(model.architecture)

# state = zero_state(2) |> chain(2, put(1=>Ry(π)))

state = run_OAA(model)

println(state)

# plotmeasure(model)