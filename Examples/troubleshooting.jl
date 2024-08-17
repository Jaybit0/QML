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

# the following code is used to troubleshoot/visualize the circuits

function build_circuit()
    rotation_precision = 2
    training_data = [[1,1]]
    model = create_oaa_circuit(training_data, rotation_precision)

    return model
end

skeleton = build_circuit()

# CURRENT STATIC 
n = 1 # number of data points
b = 2 # length of training data

## START -- run_oaa troubleshooting -- ##
models = skeleton.models
transitions = skeleton.transition_models

iter = skeleton.num_bits

# set up initial state
n = skeleton.total_num_lanes;

# define R0lstar
# TODO: fix mismatch of lanes
R0lstar = chain(
    skeleton.num_bits - 2 + 2 * skeleton.rotation_precision + 1,
    repeat(X, skeleton.num_bits + 2:skeleton.num_bits + skeleton.rotation_precision + 1),
    cz(skeleton.num_bits + 2:skeleton.num_bits + skeleton.rotation_precision, skeleton.num_bits + skeleton.rotation_precision + 1),
    repeat(X, skeleton.num_bits + 2:skeleton.num_bits + skeleton.rotation_precision + 1),
);

circuit = chain(n);

for i in 1:iter
    # organize lanes
    target_lane = models[i].global_lane_map.target_lane;
    # get RxChain and lanes
    collected_rx_lanes = vcat([target_lane], models[i].global_lane_map.rx_model_lanes, models[i].global_lane_map.rx_param_lanes);
    # get RyChain and lanes
    collected_ry_lanes = vcat([target_lane], models[i].global_lane_map.ry_model_lanes, models[i].global_lane_map.ry_param_lanes);

    circuit = chain(n, 
        subroutine(circuit, 1:n),
        subroutine(models[i].rx_compiled_architecture, collected_rx_lanes),
        subroutine(models[i].ry_compiled_architecture, collected_ry_lanes)
    );

    # circuit = chain(n, 
    #     subroutine(circuit, 1:n),
    #     subroutine(models[i].rx_compiled_architecture, collected_rx_lanes),
    #     subroutine(Daggered(models[i].rx_compiled_architecture), collected_rx_lanes),
    #     subroutine(R0lstar, collected_rx_lanes),
    #     subroutine(models[i].rx_compiled_architecture, collected_rx_lanes),
    #     subroutine(models[i].ry_compiled_architecture, collected_ry_lanes)
    # );
end

vizcircuit(circuit)
## END -- run_oaa troubleshooting -- ##