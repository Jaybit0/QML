# TODO: fix CNOT inclusion in U (remove from U, U^\dagger)

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
    training_data = [
        [1,1]
    ]
    rotation_precision = 1
    model = create_oaa_circuit(training_data, rotation_precision);

    return model
end

skeleton = build_circuit();
b = 2
rotation_precision = 1

# CURRENT STATIC 
## START -- learn_distribution -- ##
## START -- run_oaa troubleshooting -- ##

b = skeleton.num_bits

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
# vizcircuit(circuit)
circuit_with_oaa = chain(skeleton.total_num_lanes);

for i in 1:b
    u_model = skeleton.architecture_list[i]["U"]
    cnot_model = skeleton.architecture_list[i]["CNOT"]
    if i != b
        transition_model = skeleton.architecture_list[i]["TRANSITION"]
    end

    # organize lanes
    target_lane = u_model.global_lane_map.target_lane;

    # get RxChain and lanes
    # collected_rx_lanes = vcat([target_lane], models[i].global_lane_map.rx_model_lanes, models[i].global_lane_map.rx_param_lanes);
    collected_rx_lanes = vcat([target_lane], u_model.global_lane_map.rx_model_lanes, u_model.global_lane_map.rx_param_lanes);
    
    # get RyChain and lanes
    # collected_ry_lanes = vcat([target_lane], models[i].global_lane_map.ry_model_lanes, models[i].global_lane_map.ry_param_lanes);
    collected_ry_lanes = vcat([target_lane], u_model.global_lane_map.ry_model_lanes, u_model.global_lane_map.ry_param_lanes);

    # RX Rotations
    circuit_with_oaa = chain(skeleton.total_num_lanes, 
        subroutine(circuit_with_oaa, 1:skeleton.total_num_lanes),
        subroutine(u_model.rx_compiled_architecture, collected_rx_lanes),
        subroutine(cnot_model.architecture, compile_lane_map(cnot_model))
    );

    # OAA on RX Rotations
    circuit_with_oaa = chain(skeleton.total_num_lanes,
        subroutine(circuit_with_oaa, 1:skeleton.total_num_lanes),
        Measure(skeleton.total_num_lanes, locs=target_lane), # DONE: fix measurement loc
        subroutine(u_model.rx_compiled_architecture', collected_rx_lanes),
        subroutine(R0lstar, collected_rx_lanes),
        subroutine(u_model.rx_compiled_architecture, collected_rx_lanes)
    );

    # OAA on Ry
    # subroutine(models[i].ry_compiled_architecture, collected_ry_lanes)

    if i != b
        transition_lane_map = compile_lane_map(transition_model);

        circuit_with_oaa = chain(skeleton.total_num_lanes,
            subroutine(circuit_with_oaa, 1:skeleton.total_num_lanes),
            subroutine(transition_model.architecture, transition_lane_map)
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

x = measure(state, nshots=100);

# store the parameter lanes to be accessed
param_lanes = Vector{Int64}()

for i in 1:length(models)
    m = models[i]
    append!(param_lanes, m.global_lane_map.rx_param_lanes)
    append!(param_lanes, m.global_lane_map.ry_param_lanes)
end

# create an empty vector to store param results
measured_params = Vector{Vector{Int64}}()

# iterate through and store parameter results
for i in 1:length(x)
    push!(measured_params, x[i][param_lanes])
end

println(measured_params)
## END -- learn_distribution -- ##

hypothesis = get_hypothesis(measured_params, rotation_precision, b)

