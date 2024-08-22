# TODO: Simple example for debugging with only one qubit, one Rx rotation, and then only Ry rotation.
# TODO: from there, move to the cascading case
# TODO: check how focus! and relax! affects the results

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
rotation_precision = 2

training_data = [[1,1], [1, 1], [1, 1]]
# training_data = [[0,0, 1],[1,0, 0], [1,1, 0]]

model = create_oaa_circuit(training_data, rotation_precision);

vizcircuit(model.architecture)

# TODO: fix register mismatch
measured_params = learn_distribution(model)

n = length(training_data) # number data points
b = length(training_data[1]) # number bits

hypothesis = get_hypothesis(measured_params, rotation_precision, b)

plotmeasure(hypothesis)

# -- START: viz oaa circuit --

skeleton = model

models = skeleton.models
transitions = skeleton.transition_models

R0lstar = chain(
    skeleton.num_bits - 2 + 2 * skeleton.rotation_precision + 1,
    repeat(X, skeleton.num_bits + 2:skeleton.num_bits + skeleton.rotation_precision + 1),
    cz(skeleton.num_bits + 2:skeleton.num_bits + skeleton.rotation_precision, skeleton.num_bits + skeleton.rotation_precision + 1),
    repeat(X, skeleton.num_bits + 2:skeleton.num_bits + skeleton.rotation_precision + 1),
);

lanes = vcat()

n = skeleton.total_num_lanes;

vizcircuit(R0lstar)

temp = chain(n,
    subroutine(R0lstar, 1:n),
    subroutine(Daggered(models[1].rx_compiled_architecture), 1:n),
    subroutine(R0lstar, 1:n),
    subroutine(models[1].rx_compiled_architecture, 1:n)
)
# -- END: viz oaa circuit