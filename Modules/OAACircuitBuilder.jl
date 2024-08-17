using Yao

export MAX_ROTATION;

export LaneMap;
export GlobalLaneMap;
export LocalLaneMap;
export TransitionLaneMap;

export CustomBlock;
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
export run_oaa;

# stores indices of parameter, model, and target lanes.
abstract type LaneMap end

# TODO: rename to more descriptive data structure 
abstract type CustomBlock end

# organizes indices of different lanes (param, model, target) 
# relative to the entire circuit
mutable struct GlobalLaneMap<:LaneMap
    size::Int # total number of lanes in the map
    lanes::Vector{Int} # range of lanes used
    rx_model_lanes::Vector{Int} # lanes used only in the Rx model 
    ry_model_lanes::Vector{Int} # lanes used only in the Ry model
    target_lane::Int # index of target lane
    rx_param_lanes::Vector{Int} # lanes used only for parameters in the Rx training
    ry_param_lanes::Vector{Int} # lanes used only for parameters in the Ry training
    cnot_param_lane::Int
end

# organizes indices of different lanes (param, model, target) 
# relative to the subsection of the circuit that corresponds to the bit
mutable struct LocalLaneMap<:LaneMap
    size::Int # total number of lanes in the map
    lanes::Vector{Int} # range of lanes used
    rx_model_lanes::Vector{Int} # lanes used only in the Rx model 
    ry_model_lanes::Vector{Int} # lanes used only in the Ry model
    target_lane::Int # index of target lane
    rx_param_lanes::Vector{Int} # lanes used only for parameters in the Rx training
    ry_param_lanes::Vector{Int} # lanes used only for parameters in the Ry training
    cnot_param_lane::Int
end

# lane mapping for transitions between bits
mutable struct TransitionLaneMap<:LaneMap
    size::Int # number of lanes involved
    lanes::Vector{Int}
end

# circuit corresponding to each bit
mutable struct ModelBlock<:CustomBlock
    architecture::ChainBlock # circuit corresponding to the specified bit
    rx_compiled_architecture::ChainBlock # circuit corresponding to only the Rx model, used in training
    ry_compiled_architecture::ChainBlock # circuit corresponding to only the Ry model, used in training
    bit::Int # index of the bit
    rotation_precision::Int
    local_lane_map::LaneMap
    global_lane_map::LaneMap
end

# circuit corresponding to transitions between bits
mutable struct TransitionBlock<:CustomBlock
    architecture::ChainBlock # circuit corresponding to CCNOT gate used to transition between bits, used in training
    global_lane_map::LaneMap
    bit::Int # index of preceding bit
end

# the complete, compiled circuit with all bits and transitions
mutable struct OAABlock<:CustomBlock
    architecture::CompositeBlock # complete circuit containing all blocks, used in visualization
    models::Vector{ModelBlock} # stores models corresponding to each bit, used in training
    transition_models::Vector{TransitionBlock} # stores models corresponding to each transition between bits, used in training
    rotation_precision::Int
    num_bits::Int # number of bits in the training data
    num_training_data::Int
    total_num_lanes::Int
end

const MAX_ROTATION = 2*π

function compile_lane_map(model::ModelBlock, b::Int)
    map = model.global_lane_map
    if model.bit == b
        return vcat(
            map.rx_model_lanes, # only one model lane
            [map.target_lane],
            map.rx_param_lanes,
            map.ry_param_lanes
        )
    else
        return vcat(
            map.rx_model_lanes, # only one model lane
            [map.target_lane],
            map.rx_param_lanes,
            map.ry_param_lanes,
            [map.cnot_param_lane]
        )
    end
end

function compile_lane_map(model::TransitionBlock)
    return model.global_lane_map.lanes
end

function compile_lane_map(model::TransitionBlock, b::Int)
    return compile_lane_map(model)
end

    # TODO: put the data point: first, second, third... bit lanes next to each other
# maps the model index to the set of lanes
# called in build_model
function map_global_lanes(bit::Int, rp::Int, n::Int, b::Int)
    # global lane locations for bit, b = total bits, n = number elements training data
    #   rp x-model bits n*(bit - 1) + 1:n*(bit - 1) + n
    #   rp y-model bits n*(bit - 1) + 1:n*(bit - 1) + n
    #   1 target bit n*b + bit
    #   rp x-parameter bits n*(b + 1) + 2*rp*(bit - 1) + 1:n*(b + 1) + 2*rp*(bit - 1) + rp
    #   rp y-parameter bits n*(b + 1) + 2*rp*(bit - 1) + rp + 1:n*(b + 1) + 2*rp*(bit - 1) + 2*rp
    #   CNOT control parameter n*(b+1)+2*rp*b + bit
    
    # TODO: clean up redundant information?
    if bit == b
        return GlobalLaneMap(
            (n + 1 + 2*rp)*b + b - 1, # size of entire circuit
            1:n*(b+1)+2*rp*b + b, # lanes
            n*(bit - 1) + 1:n*bit, # rp x-model
            n*(bit - 1) + 1:n*bit, # rp y-model
            n*b + bit, # target bit
            (n + 1)*b + rp*(bit - 1) + 1:(n + 1)*b + rp*(bit), # rp x-param
            (n + 1 + rp)*b + rp*(bit - 1) + 1:(n + 1 + rp)*b + rp * (bit), # rp y-param
            -1 # no CNOT
        )
    else
        return GlobalLaneMap(
            (n + 1 + 2*rp)*b + b - 1, # size of entire circuit
            1:n*(b+1)+2*rp*b + b, # lanes
            n*(bit - 1) + 1:n*bit, # rp x-model
            n*(bit - 1) + 1:n*bit, # rp y-model
            n*b + bit, # target bit
            (n + 1)*b + rp*(bit - 1) + 1:(n + 1)*b + rp*(bit), # rp x-param
            (n + 1 + rp)*b + rp*(bit - 1) + 1:(n + 1 + rp)*b + rp * (bit), # rp y-param
            (n + 1 + 2*rp)*b + bit # cnot param lane
        )
    end
end

# TODO: remove local lane map references
# called in build_model
function map_local_lanes(n::Int, b::Int, rp::Int, bit::Int)
    # local lane locations:
    # total = 
    #   rp x-model bits 1:n
    #   rp y-model bits 1:n
    #   1 target bit n + 1
    #   rp x-parameter bits n + 2:n + 1 + rp
    #   rp y-parameter bits n + 2 + rp:n + 1 + 2rp
    #   + 1 CNOT control parameter n + 2 + 2rp

    if bit == b
        return LocalLaneMap(
            n + 1 + 2*rp, # size of ONLY lanes applicable to this bit
            1:n + 1 + 2*rp, # all lanes
            1:n, # rx model
            1:n, # ry model
            n + 1, # target
            n + 2:n + 1 + rp, # rx param
            n + 2 + rp:n + 1 + 2*rp, # ry param
            -1 # no CNOT
        )
    else
        return LocalLaneMap(
            n + 2 + 2*rp, # size of ONLY lanes applicable to this bit
            1:n + 2 + 2*rp, # all lanes
            1:n, # rx model
            1:n, # ry model
            n + 1, # target
            n + 2:n + 1 + rp, # rx param
            n + 2 + rp:n + 1 + 2*rp, # ry param
            n + 2 + 2*rp # CNOT
        )
    end
end

# returns the lanes for the transition circuit corresponding to the given bits
# called in build_transition
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

# TODO: clean up extra lane at the end
# builds the model for an individual qubit
# called by create_oaa_circuit
function build_model(bit::Int, rotation_precision::Int, training_data::Vector{Vector{Int}})
    # TODO: implement checks on Vector sizes
    n = length(training_data)
    b = length(training_data[1])

    local_lanes = map_local_lanes(n, b, rotation_precision, bit)

    # controlled rx rotations
    ctrl_rotx(ctrl, target, θ) = control(ctrl, target => Rx(θ))

    # controlled ry rotations
    ctrl_roty(ctrl, target, θ) = control(ctrl, target => Ry(θ))

    # model
    rx_subchain = chain(rotation_precision + 1, ctrl_rotx(j + 1, 1, MAX_ROTATION / 2^(j)) for j in 1:rotation_precision);

    x_temp = chain(
        rotation_precision + n,
        subroutine(rx_subchain, pushfirst!(collect(n + 1:n + rotation_precision), i)) for i in 1:n
    );

    rx_chain = chain(
        rotation_precision + n,
        repeat(H, n+1:n+rotation_precision),
        subroutine(x_temp, 1:rotation_precision + n)
    );

    # TODO: fix angle increment
    ry_subchain = chain(rotation_precision + 1, ctrl_roty(j + 1, 1, MAX_ROTATION / 2^(j)) for j in 1:rotation_precision);

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

    rx_compiled_architecture = chain(
        1 + n + rotation_precision,
        subroutine(rx_chain, 2:1 + n + rotation_precision),
        subroutine(cnot_subblock, push!(collect(2:n+1), 1))
    )

    ry_compiled_architecture = chain(
        1 + n + rotation_precision,
        subroutine(ry_chain, 2:1 + n + rotation_precision),
        subroutine(cnot_subblock, push!(collect(2:n+1), 1))
    )

    
    return ModelBlock(
        model,
        rx_compiled_architecture,
        ry_compiled_architecture,
        bit,
        rotation_precision,
        local_lanes,
        global_lanes
    )
end

# TODO: fix reliance on ctrl from other method
# called by create_oaa_circuit
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

# generates the underlying cascading circuit for oblivious amplitude amplitification
# to run OAA on the result, use run_OAA()
function create_oaa_circuit(training_data::Vector{Vector{Int}}, rotation_precision::Int)
    if length(training_data) < 1
        # TODO: implement checks
        return nothing
    end

    b = length(training_data[1]) # number bits per data point
    n = length(training_data) # total number of data points

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

    # compile final architecture
    architecture = chain(
        models[1].global_lane_map.size,
        subroutine(model.architecture, compile_lane_map(model, b)) for model in architecture_list
    )

    return OAABlock(
        architecture,
        models,
        transitions,
        rotation_precision,
        b,
        n,
        models[1].global_lane_map.size
    );
end

# compiles and runs the given circuit using OAA
# by iterating over each of the subblocks for bit-by-bit training
# e.g. training based on the given circuit
function run_oaa(skeleton::OAABlock)
    models = skeleton.models
    transitions = skeleton.transition_models

    iter = skeleton.num_bits

    # set up initial state

    state = zero_state(skeleton.total_num_lanes);

    # define R0lstar
    # TODO: fix mismatch of lanes
    R0lstar = chain(
        skeleton.num_training_data + skeleton.rotation_precision + 1,
        repeat(X, skeleton.num_bits + 2:skeleton.num_bits + skeleton.rotation_precision + 1),
        cz(skeleton.num_bits + 2:skeleton.num_bits + skeleton.rotation_precision, skeleton.num_bits + skeleton.rotation_precision + 1),
        repeat(X, skeleton.num_bits + 2:skeleton.num_bits + skeleton.rotation_precision + 1),
    );

    # iterate over the subblocks to train bit-by-bit
    for i in 1:iter
        # organize lanes
        target_lane = models[i].global_lane_map.target_lane;

        ## get RxChain and lanes
        collected_rx_lanes = vcat([target_lane], models[i].global_lane_map.rx_model_lanes, models[i].global_lane_map.rx_param_lanes);
        # collected_rx_lanes = vcat([target_lane], models[i].global_lane_map.rx_param_lanes);

        ## RyChain and lanes
        collected_ry_lanes = vcat([target_lane], models[i].global_lane_map.ry_model_lanes, models[i].global_lane_map.ry_param_lanes);
        # collected_ry_lanes = vcat([target_lane], models[i].global_lane_map.ry_param_lanes);

        # run state through first model
        ## focus Rx lanes
        focus!(state, collected_rx_lanes);

        ## pipe state into RxChain
        state |> models[i].rx_compiled_architecture;

        ## -- TROUBLESHOOTING CODE -- 
        println(models[i].global_lane_map.rx_param_lanes)
        println(collected_rx_lanes)
        println(skeleton.num_training_data + 2 * skeleton.rotation_precision + 1)
        ## -- TROUBLESHOOTING CODE -- 
            
        ## measure outcome
        outcome = measure!(state, 1);

        ## if outcome != 0, run OAA again
        if outcome != 0
            state |> Daggered(models[i].rx_compiled_architecture);
            state |> R0lstar;
            state |> models[i].rx_compiled_architecture;
        end

        ## relax Rx lanes
        relax!(state, collected_rx_lanes);

        ## focus Ry lanes
        focus!(state, collected_ry_lanes);

        ## pipe state into RyChain
        state |> models[i].ry_compiled_architecture;

        ## measure outcome
        outcome = measure!(state, 1)

        ## if outcome != 0, run OAA again
        if outcome != 0
            state |> Daggered(models[i].ry_compiled_architecture);
            state |> R0lstar;
            state |> models[i].ry_compiled_architecture;
        end

        ## relax Ry lanes
        relax!(state, collected_ry_lanes);

        # if i != iter
            # append the transition model
        if i != iter
            transition_lane_map = compile_lane_map(transitions[i])
            focus!(state, transition_lane_map)
            state |> transitions[i].architecture
            relax!(state, transition_lane_map)
        end
    end

    return state
end