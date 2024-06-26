using Yao

export MAX_ROTATION;

export LaneMap;
export GlobalLaneMap;
export LocalLaneMap;
export ModelBlock;
export OAABlock;

export compile_lane_map;
export map_global_lanes;
export map_local_lanes;
export build_model;
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
end

const MAX_ROTATION = 2*π

function compile_lane_map(map::LaneMap)
    # lanes::Vector{Int}
    # rx_model_lanes::Vector{Int}
    # ry_model_lanes::Vector{Int}
    # target_lane::Int
    # rx_param_lanes::Vector{Int}
    # ry_param_lanes::Vector{Int}
    # cnot_param_lane::Int
    out = vcat(
        map.rx_model_lanes, # only one model lane
        [map.target_lane],
        map.rx_param_lanes,
        map.ry_param_lanes,
        [map.cnot_param_lane]
    )
    return out
end

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

# builds the model for an individual qubit
function build_model(bit::Int, rotation_precision, training_data::Vector{Vector{Int}})
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

    # TODO: model
    rx_subchain = chain(rotation_precision + 1, ctrl_rotx(j + 1, 1, rotation_increment * j) for j in 1:rotation_precision);

    x_temp = chain(
        rotation_precision + n,
        subroutine(rx_subchain, pushfirst!(collect(n + 1:n + rotation_precision), i)) for i in 1:n
    )

    rx_chain = chain(
        rotation_precision + n,
        repeat(H, n+1:n+rotation_precision),
        subroutine(x_temp, 1:rotation_precision + n)
    )

    ry_subchain = chain(rotation_precision + 1, ctrl_roty(j + 1, 1, rotation_increment * j) for j in 1:rotation_precision);

    y_temp = chain(
        rotation_precision + n,
        subroutine(ry_subchain, pushfirst!(collect(n + 1:n + rotation_precision), i)) for i in 1:n
    )

    ry_chain = chain(
        rotation_precision + n,
        repeat(H, n+1:n+rotation_precision),
        subroutine(y_temp, 1:rotation_precision + n)
    )

    model = chain(
        local_lanes.size,
        subroutine(rx_chain, vcat(local_lanes.rx_model_lanes, local_lanes.rx_param_lanes)),
        subroutine(ry_chain, vcat(local_lanes.ry_model_lanes, local_lanes.ry_param_lanes))
    )

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
    )

    model = chain(
        local_lanes.size,
        subroutine(model, local_lanes.lanes),
        subroutine(x_from_data, local_lanes.lanes)
    )

    cnot_lanes = vcat(1:n, [local_lanes.target_lane])

    cnot_subblock = chain(
        n + 1,
        cnot(1:n, n+1)
    )

    model = chain(
        local_lanes.size,
        put(local_lanes.lanes => model),
        control(
            local_lanes.size,
            local_lanes.cnot_param_lane,
            cnot_lanes => cnot_subblock
        )
    )

    global_lanes = map_global_lanes(bit, rotation_precision, n, b)

    
    return ModelBlock(
        model,
        bit,
        rotation_precision,
        local_lanes,
        global_lanes
    )
end

function create_oaa_circuit(training_data::Vector{Vector{Int}}, rotation_precision::Int)
    if length(training_data) < 1
        # TODO: throw error
        return nothing
    end

    b = length(training_data[1])
    n = length(training_data)

    models = []

    for bit in 1:b
        model = build_model(bit, rotation_precision, training_data)
        push!(models, model)
    end

    architecture = chain(
        models[1].global_lane_map.size,
        subroutine(model.architecture, compile_lane_map(model.global_lane_map)) for model in models
    )

    return OAABlock(
        architecture,
        models,
        rotation_precision
    )
end