using Yao

export MAX_ROTATION;

export LaneMap;
export GlobalLaneMap;
export LocalLaneMap;
export TransitionLaneMap;
export TransitionBlock;
export ModelBlock;
export OAABlock;

export compile_lane_map;
export map_global_lanes;
export map_local_lanes;
export map_transition_lanes;
export build_model;
export build_transition;
export create_oaa_circuit;

abstract type LaneMap end

mutable struct GlobalLaneMap<:LaneMap
    size::Int
    lanes::Vector{Int}
    rx_model_lanes::Vector{Int}
    ry_model_lanes::Vector{Int}
    target_lane::Int
    rx_param_lanes::Vector{Int}
    ry_param_lanes::Vector{Int}
    cnot_param_lane::Int
end

mutable struct LocalLaneMap<:LaneMap
    size::Int
    lanes::Vector{Int}
    rx_model_lanes::Vector{Int}
    ry_model_lanes::Vector{Int}
    target_lane::Int
    rx_param_lanes::Vector{Int}
    ry_param_lanes::Vector{Int}
    cnot_param_lane::Int
end

mutable struct TransitionLaneMap<:LaneMap
    size::Int
    lanes::Vector{Int}
end

mutable struct ModelBlock
    architecture::ChainBlock
    bit::Int
    rotation_precision::Int
    local_lane_map::LaneMap
    global_lane_map::LaneMap
end

mutable struct OAABlock
    architecture::CompositeBlock
    models::Vector{ModelBlock}
    rotation_precision::Int
    total_num_lanes::Int
end

mutable struct TransitionBlock
    architecture::ChainBlock
    global_lane_map::LaneMap
    bit::Int
end

const MAX_ROTATION = 2*π

function compile_lane_map(map::GlobalLaneMap)
    # lanes::Vector{Int}
    # rx_model_lanes::Vector{Int}
    # ry_model_lanes::Vector{Int}
    # target_lane::Int
    # rx_param_lanes::Vector{Int}
    # ry_param_lanes::Vector{Int}
    # cnot_param_lane::Int
    return vcat(
            map.rx_model_lanes, # only one model lane
            [map.target_lane],
            map.rx_param_lanes,
            map.ry_param_lanes,
            [map.cnot_param_lane]
        )
    end

function compile_lane_map(map::TransitionLaneMap)
    return map.lanes
end

    # TODO: put the data point: first, second, third... bit lanes next to each other
# maps the model index to the set of lanes
function map_global_lanes(bit::Int, rp::Int, n::Int, b::Int)
    # global lane locations for bit, b = total bits, n = number elements training data
    #   rp x-model bits n*(bit - 1) + 1:n*(bit - 1) + n
    #   rp y-model bits n*(bit - 1) + 1:n*(bit - 1) + n
    #   1 target bit n*b + bit
    #   rp x-parameter bits n*(b + 1) + 2*rp*(bit - 1) + 1:n*(b + 1) + 2*rp*(bit - 1) + rp
    #   rp y-parameter bits n*(b + 1) + 2*rp*(bit - 1) + rp + 1:n*(b + 1) + 2*rp*(bit - 1) + 2*rp
    #   CNOT control parameter n*(b+1)+2*rp*b + bit
    
    return GlobalLaneMap(
        n*(b+1)+2*rp*b + b, # size,
        1:n*(b+1)+2*rp*b + b, # lanes
        n*(bit - 1) + 1:n*(bit - 1) + n, # rp x-model
        n*(bit - 1) + 1:n*(bit - 1) + n, # rp y-model
        n*b + bit, # target bit
        n*(b + 1) + 2*rp*(bit - 1) + 1:n*(b + 1) + 2*rp*(bit - 1) + rp, # rp x-param
        n*(b + 1) + 2*rp*(bit - 1) + rp + 1:n*(b + 1) + 2*rp*(bit - 1) + 2*rp, # rp y-param
        n*(b+1)+2*rp*b + bit # cnot param lane
    )
end


function map_local_lanes(n::Int, rp::Int)
    # local lane locations:
    # total = 
    #   rp x-model bits 1:n
    #   rp y-model bits 1:n
    #   1 target bit n + 1
    #   rp x-parameter bits n + 2:n + 1 + rp
    #   rp y-parameter bits n + 2 + rp:n + 1 + 2rp
    #   + 1 CNOT control parameter n + 2 + 2rp

    return LocalLaneMap(
        n + 2 + 2*rp, # size
        1:n + 2 + 2*rp, # all lanes
        1:n, # rx model
        1:n, # ry model
        n + 1, # target
        n + 2:n + 1 + rp, # rx param
        n + 2 + rp:n + 1 + 2*rp, # ry param
        n + 2 + 2*rp # CNOT
    )
end

function map_transition_lanes(bit::Int, n::Int, ctrl::Int)
    # bit = bit number
    # n = number elements training data
    # ctrl = control parameter
    lanes = collect((bit - 1) * n + 1:(bit + 1) * n)

    push!(lanes, ctrl)
    
    return TransitionLaneMap(
        2*n + 1,
        lanes
    )
end

# builds the model for an individual qubit
function build_model(bit::Int, rotation_precision::Int, training_data::Vector{Vector{Int}})
    # TODO: implement checks on Vector sizes
    n = length(training_data)
    b = length(training_data[1])

    local_lanes = map_local_lanes(n, rotation_precision)

    # DONE: model bits
    rotation_increment = MAX_ROTATION / (rotation_precision + 1)

    # DONE: controlled rx rotations
    ctrl_rotx(ctrl, target, θ) = control(ctrl, target => Rx(θ))

    # DONE: controlled ry rotations
    ctrl_roty(ctrl, target, θ) = control(ctrl, target => Ry(θ))

    # DONE: model
    rx_subchain = chain(rotation_precision + 1, ctrl_rotx(j + 1, 1, rotation_increment * j) for j in 1:rotation_precision);

    x_temp = chain(
        rotation_precision + n,
        subroutine(rx_subchain, pushfirst!(collect(n + 1:n + rotation_precision), i)) for i in 1:n
    );

    rx_chain = chain(
        rotation_precision + n,
        repeat(H, n+1:n+rotation_precision),
        subroutine(x_temp, 1:rotation_precision + n)
    );

    ry_subchain = chain(rotation_precision + 1, ctrl_roty(j + 1, 1, rotation_increment * j) for j in 1:rotation_precision);

    y_temp = chain(
        rotation_precision + n,
        subroutine(ry_subchain, pushfirst!(collect(n + 1:n + rotation_precision), i)) for i in 1:n
    );

    ry_chain = chain(
        rotation_precision + n,
        repeat(H, n+1:n+rotation_precision),
        subroutine(y_temp, 1:rotation_precision + n)
    );

    model = chain(
        local_lanes.size,
        subroutine(rx_chain, vcat(local_lanes.rx_model_lanes, local_lanes.rx_param_lanes)),
        subroutine(ry_chain, vcat(local_lanes.ry_model_lanes, local_lanes.ry_param_lanes))
    );

    # DONE: X gates
    zeros = Int[]
    for i in 1:n
        if training_data[i][bit] == 0
            push!(zeros, i)
        end
    end

    x_from_data = chain(
        local_lanes.size,
        put(i => X) for i in zeros
    );

    model = chain(
        local_lanes.size,
        subroutine(model, local_lanes.lanes),
        subroutine(x_from_data, local_lanes.lanes)
    );

    cnot_lanes = vcat(1:n, [local_lanes.target_lane])

    cnot_subblock = chain(
        n + 1,
        cnot(1:n, n+1)
    );

    model = chain(
        local_lanes.size,
        put(local_lanes.lanes => model),
        subroutine(cnot_subblock, cnot_lanes)
    );

    global_lanes = map_global_lanes(bit, rotation_precision, n, b)

    
    return ModelBlock(
        model,
        bit,
        rotation_precision,
        local_lanes,
        global_lanes
    )
end

# TODO: fix reliance on ctrl from other method
function build_transition(bit::Int, ctrl_index::Int, n::Int)

    lanes = map_transition_lanes(bit, n, ctrl_index)

    # TODO: make nested CNOT
    cnot_subblock = control(
        3,
        3,
        1:2 => chain(2, cnot(1, 2))
    );

    # append CNOT transition gates
    model = chain(
        2 * n + 1,
        subroutine(cnot_subblock, [i, i + n, 2*n + 1]) for i in 1:n
    )

    # append H gates
    model = chain(
        2*n + 1,
        put(2*n+1=> H),
        subroutine(model, 1:2*n+1)
    )

    return TransitionBlock(
        model,
        lanes,
        bit
    )
end

function create_oaa_circuit(training_data::Vector{Vector{Int}}, rotation_precision::Int)
    if length(training_data) < 1
        # TODO: implement checks
        return nothing
    end

    b = length(training_data[1])
    n = length(training_data)

    models = []
    transitions = []
    architecture_list = []

    for bit in 1:b
        model = build_model(bit, rotation_precision, training_data)
        push!(models, model)
        push!(architecture_list, model)
        if bit != b
            transition = build_transition(bit, model.global_lane_map.cnot_param_lane, n)
            push!(architecture_list, transition)
            push!(transitions, transition)
        end
    end;

    # TODO: implement OAA here
    architecture = chain(
        models[1].global_lane_map.size,
        subroutine(model.architecture, compile_lane_map(model.global_lane_map)) for model in architecture_list
        # TODO: add CNOT controls
    )

    transition_architecture = chain(
        models[1].global_lane_map.size,
        subroutine(models[1].global_lane_map.size, transitions[1].architecture, push!(collect(1:6), 14))
    )

    return OAABlock(
        architecture,
        models,
        rotation_precision,
        models[1].global_lane_map.size
    );
end