# ======== IMPORTS ========
# =========================

if !isdefined(Main, :QML)
    include("../Modules/OAACircuitBuilder.jl")
    using .QML
end

using Yao
using Yao.EasyBuild, YaoPlots

param_lanes = 1:3
num_model_lanes = 1

model_architecture = create_OAACircuit(param_lanes, num_model_lanes)

vizcircuit(model_architecture)