# ======== IMPORTS ========
# =========================
include("../Modules/OAACircuitBuilder.jl")

using Yao
using Yao.EasyBuild, YaoPlots


model_lanes = 1:2
rotation_precision = 2

model_architecture = create_OAACircuit(model_lanes, rotation_precision)

vizcircuit(model_architecture)