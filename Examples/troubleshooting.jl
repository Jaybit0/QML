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
    training_data = [[1,1]]
    rotation_precision = 1
    model = create_oaa_circuit(training_data, rotation_precision);

    return model
end

skeleton = build_circuit();

# CURRENT STATIC 
## START -- learn_distribution -- ##
## START -- run_oaa troubleshooting -- ##
models = skeleton.models
transitions = skeleton.transition_models

iter = skeleton.num_bits

# set up initial state
# t = skeleton.total_num_lanes;

# define R0lstar
# TODO: fix mismatch of lanes
# notes: if num data = 1, doesn't fit
# check to see if order of applying circuit lanes is the same after focus?
R0lstar = chain(
    skeleton.num_training_data + skeleton.rotation_precision + 1,
    repeat(X, skeleton.num_training_data + 2:skeleton.num_training_data + skeleton.rotation_precision + 1),
    cz(skeleton.num_training_data + 2:skeleton.num_training_data + skeleton.rotation_precision, skeleton.num_training_data + skeleton.rotation_precision + 1),
    repeat(X, skeleton.num_training_data + 2:skeleton.num_training_data + skeleton.rotation_precision + 1),
);

circuit = chain(skeleton.total_num_lanes);
circuit_with_oaa = chain(skeleton.total_num_lanes);

for i in 1:iter
    # organize lanes
    target_lane = models[i].global_lane_map.target_lane;
    # get RxChain and lanes
    collected_rx_lanes = vcat([target_lane], models[i].global_lane_map.rx_model_lanes, models[i].global_lane_map.rx_param_lanes);
    # get RyChain and lanes
    collected_ry_lanes = vcat([target_lane], models[i].global_lane_map.ry_model_lanes, models[i].global_lane_map.ry_param_lanes);

    # RX Rotations
    circuit_with_oaa = chain(skeleton.total_num_lanes, 
        subroutine(circuit_with_oaa, 1:skeleton.total_num_lanes),
        subroutine(models[i].rx_compiled_architecture, collected_rx_lanes)
    );
    # OAA on RX Rotations
    circuit_with_oaa = chain(skeleton.total_num_lanes,
        subroutine(circuit_with_oaa, 1:skeleton.total_num_lanes),
        Measure(skeleton.total_num_lanes, locs=1), # TODO: fix measurement loc
        subroutine(models[i].rx_compiled_architecture', collected_rx_lanes),
        subroutine(R0lstar, collected_rx_lanes),
        subroutine(models[i].rx_compiled_architecture, collected_rx_lanes)
    );

    # subroutine(models[i].ry_compiled_architecture, collected_ry_lanes)

    if i != iter
        transition_lane_map = compile_lane_map(transitions[i]);

        circuit_with_oaa = chain(skeleton.total_num_lanes,
            subroutine(circuit_with_oaa, 1:skeleton.total_num_lanes),
            subroutine(transitions[i].architecture, transition_lane_map)
        )
    end
end

vizcircuit(circuit_with_oaa)

# get blank circuit without any OAA
circuit = skeleton.architecture
vizcircuit(circuit)
## END -- run_oaa troubleshooting -- ##

state = zero_state(skeleton.total_num_lanes)

state |> circuit_with_oaa

x = measure()

## END -- learn_distribution -- ##