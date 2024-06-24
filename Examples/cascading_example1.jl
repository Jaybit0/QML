# ======== IMPORTS ========
# =========================
include("../Modules/OAACircuitBuilder.jl")

using Yao
using Yao.EasyBuild, YaoPlots

num_model_lanes = 2
rotation_precision = 1

ma = create_OAACircuit(num_model_lanes, rotation_precision)

vizcircuit(ma.architecture)

Φ = zero_state(2) |> chain(2, put(1=>Ry(π)))
run_OAA(ma, Φ)



