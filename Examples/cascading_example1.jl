# ======== IMPORTS ========
# =========================
if !isdefined(Main, :QML)
    include("../Modules/QML.jl")
    using .QML
end
include("../Modules/OAACircuitBuilder.jl")
include("../Modules/OAAPlotting.jl")

using Yao
using Yao.EasyBuild, YaoPlots

# num_model_lanes = 2
rotation_precision = 3

training_data = [[0,0],[1,0], [0, 0], [1, 0]]
# training_data = [[0,0, 1],[1,0, 0], [1,1, 0]]

model = create_oaa_circuit(training_data, rotation_precision)

vizcircuit(model.architecture)

measured_params = learn_distribution(model)

n = 2 # number data points
b = 2 # number bits

hypothesis = get_hypothesis(measured_params, rotation_precision, b)

plotmeasure(hypothesis)