#################
# Undergoing revisions:
# TODO: fix the level at which n, b are accessed
# TODO: change organization to be RX_U, RX_CNOT, RY_U, RY_CNOT
# TODO: refactor code
#################

using Yao

export MAX_ROTATION;

export AbstractLaneMap;
export GlobalLaneMap;
export LocalLaneMap;
export TransitionLaneMap;

export AbstractCustomBlock;
export TransitionBlock;
export ModelBlock;
export OAABlock;

export compile_lane_map;
export map_global_lanes;
export map_local_lanes;
export map_transition_lanes;
export build_U;
export build_transition;
export create_oaa_circuit;
export run_oaa;

# stores indices of parameter, model, and target lanes.
abstract type AbstractLaneMap end

abstract type AbstractCustomBlock end


# # organizes indices of different lanes (param, model, target) 
# # relative to the entire circuit
mutable struct LaneMap<:AbstractLaneMap
    size::Int
    lanes::Vector{Int} # range of lanes used
    rx_model_lanes::Vector{Int} # lanes used only in the Rx model 
    ry_model_lanes::Vector{Int} # lanes used only in the Ry model
    rx_target_lane::Int # index of target lane in rx rotations
    ry_target_lane::Int # index of target lane in ry rotations
    rx_param_lanes::Vector{Int} # lanes used only for parameters in the Rx training
    ry_param_lanes::Vector{Int} # lanes used only for parameters in the Ry training
    cnot_param_lane::Int
end

# Simple lane map with minimal requirements
mutable struct SimpleLaneMap<:AbstractLaneMap
    size::Int
    lanes::Vector{Int}
end

# lane mapping for transitions between bits
mutable struct TransitionLaneMap<:AbstractLaneMap
    size::Int # number of lanes involved
    lanes::Vector{Int}
end

# generic custom block
mutable struct CustomBlock<:AbstractCustomBlock
    architecture::ChainBlock
    global_lane_map::AbstractLaneMap
end

# circuit corresponding to each bit
mutable struct ModelBlock<:AbstractCustomBlock
    architecture::ChainBlock # circuit corresponding to the specified bit
    rx_compiled_architecture::ChainBlock # circuit corresponding to only the Rx model, used in training
    ry_compiled_architecture::ChainBlock # circuit corresponding to only the Ry model, used in training
    bit::Int # index of the bit
    rotation_precision::Int
    local_lane_map::AbstractLaneMap
    global_lane_map::AbstractLaneMap
end

# circuit corresponding to transitions between bits
mutable struct TransitionBlock<:AbstractCustomBlock
    architecture::ChainBlock # circuit corresponding to CCNOT gate used to transition between bits, used in training
    global_lane_map::AbstractLaneMap
    bit::Int # index of preceding bit
end

# the complete, compiled circuit with all bits and transitions
mutable struct OAABlock<:AbstractCustomBlock
    architecture::CompositeBlock # complete circuit containing all blocks, used in visualization
    architecture_list::Vector{Dict{String, AbstractCustomBlock}}
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
            [map.rx_target_lane],
            [map.ry_target_lane],
            map.rx_param_lanes,
            map.ry_param_lanes
        )
    else
        return vcat(
            map.rx_model_lanes, # only one model lane
            [map.rx_target_lane],
            [map.ry_target_lane],
            map.rx_param_lanes,
            map.ry_param_lanes,
            [map.cnot_param_lane]
        )
    end
end

function compile_lane_map(model::AbstractCustomBlock)
    return model.global_lane_map.lanes
end

function compile_lane_map(model::AbstractCustomBlock, b::Int)
    return compile_lane_map(model)
end

# maps the model index to the set of lanes
# called in build_model
function map_global_lanes(bit::Int, rp::Int, n::Int, b::Int)
    # global lane locations for bit, b = total bits, n = number elements training data
    #   rp x-model bits n*(bit - 1) + 1:n*(bit - 1) + n
    #   rp y-model bits n*(bit - 1) + 1:n*(bit - 1) + n
    #   1 rx target bit n*b + bit
    #   1 ry target bit n*b + b + bit
    #   rp x-parameter bits n*(b + 1) + 2*rp*(bit - 1) + 1:n*(b + 1) + 2*rp*(bit - 1) + rp
    #   rp y-parameter bits n*(b + 1) + 2*rp*(bit - 1) + rp + 1:n*(b + 1) + 2*rp*(bit - 1) + 2*rp
    #   CNOT control parameter n*(b+1)+2*rp*b + bit
    if bit == b
        return LaneMap(
            (n + 2 + 2*rp)*b + b - 1, # size of entire circuit
            collect(1:(n + 2 + 2*rp)*b + b - 1), # lanes
            collect(n*(bit - 1) + 1:n*bit), # rp x-model
            collect(n*(bit - 1) + 1:n*bit), # rp y-model
            n*b + bit, # rx target bit
            n*b + b + bit, # ry target bit
            collect((n + 2)*b + rp*(bit - 1) + 1:(n + 2)*b + rp*(bit)), # rp x-param
            collect((n + 2 + rp)*b + rp*(bit - 1) + 1:(n + 2 + rp)*b + rp * (bit)), # rp y-param
            -1 # no CNOT
        )
    else
        return LaneMap(
            (n + 2 + 2*rp)*b + b - 1, # size of entire circuit
            collect(1:(n + 2 + 2*rp)*b + b - 1), # lanes
            collect(n*(bit - 1) + 1:n*bit), # rp x-model
            collect(n*(bit - 1) + 1:n*bit), # rp y-model
            n*b + bit, # rx target bit
            n*b + b + bit, # ry target bit
            collect((n + 2)*b + rp*(bit - 1) + 1:(n + 2)*b + rp*(bit)), # rp x-param
            collect((n + 2 + rp)*b + rp*(bit - 1) + 1:(n + 2 + rp)*b + rp * (bit)), # rp y-param
            (n + 2 + 2*rp)*b + bit # cnot param lane
        )
    end
end

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
        size = n + 2 + 2*rp
        return LaneMap(
            size, # size of ONLY lanes applicable to this bit
            collect(1:size), # all lanes
            collect(1:n), # rx model
            collect(1:n), # ry model
            n + 1, # rx target
            n + 2, # ry target
            collect(n + 2 + 1:n + 2 + rp), # rx param
            collect(n + 2 + rp + 1:n + 2 + 2*rp), # ry param
            -1 # no CNOT
        )
    else
        size = n + 2 + 2*rp + 1
        return LaneMap(
            size, # size of ONLY lanes applicable to this bit
            collect(1:size), # all lanes
            collect(1:n), # rx model
            collect(1:n), # ry model
            n + 1, # rx target
            n + 2, # ry target
            collect(n + 2 + 1:n + 2 + rp), # rx param
            collect(n + 2 + rp + 1:n + 2 + 2*rp), # ry param
            n + 2 + 2*rp + 1 # CNOT
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

# builds the CNOT model
function build_CNOT(n::Int, bit::Int, rotation_precision::Int, target_lane::Int)
    cnot_lanes = vcat(n*(bit - 1) + 1:n*bit, [target_lane])
    cnot_lane_map = SimpleLaneMap(
        n + 1,
        cnot_lanes
    )

    cnot_subblock = chain(
        n + 1,
        cnot(1:n, n + 1)
    )

    return CustomBlock(
        cnot_subblock, # architecture
        cnot_lane_map
    )
end

function build_RX_U(bit::Int, rotation_precision::Int, training_data::Vector{Vector{Int}})
    n = length(training_data)
    b = length(training_data[1])

    local_lanes = map_local_lanes(n, b, rotation_precision, bit)
    ctrl_rotx(ctrl, target, θ) = control(ctrl, target => Rx(θ))
end

function build_RX_CNOT(bit::Int, rotation_precision::Int, training_data::Vector{Vector{Int}})
end

function build_RY_U(bit::Int, rotation_precision::Int, training_data::Vector{Vector{Int}})
end

function build_RY_CNOT(bit::Int, rotation_precision::Int, training_data::Vector{Vector{Int}})
end

# builds the model for an individual qubit
# called by create_oaa_circuit
function build_U(bit::Int, rotation_precision::Int, training_data::Vector{Vector{Int}})
    n = length(training_data)
    b = length(training_data[1])

    local_lanes = map_local_lanes(n, b, rotation_precision, bit)

    # controlled rotations operations
    ctrl_rotx(ctrl, target, θ) = control(ctrl, target => Rx(θ))
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

    # apply NOT gates to places where the data has value 0
    zeros = Int[]
    for i in 1:n
        if training_data[i][bit] == 0
            push!(zeros, i)
        end
    end

    x_from_data = chain(
        n,
        put(i => X) for i in zeros
    );

    model = chain(
        local_lanes.size,
        subroutine(model, local_lanes.lanes),
        subroutine(x_from_data, 1:n)
    );

    global_lanes = map_global_lanes(bit, rotation_precision, n, b)

    rx_compiled_architecture = chain(
        1 + n + rotation_precision,
        subroutine(rx_chain, 2:1 + n + rotation_precision),
        subroutine(x_from_data, 2:1 + n)
    )

    ry_compiled_architecture = chain(
        1 + n + rotation_precision,
        subroutine(ry_chain, 2:1 + n + rotation_precision),
        subroutine(x_from_data, 2:1 + n)
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

# builds models for the specified qubit
# returns a list of models that are to be added to the final architecture
function build_model(bit::Int, rotation_precision::Int, training_data::Vector{Vector{Int}})
    b = length(training_data[1]) # number bits per data point
    n = length(training_data) # total number of data points
    
    # what is returned
    models_dict = Dict{String, AbstractCustomBlock}()

    # -- BUILD U --
    U_model = build_U(bit, rotation_precision, training_data)
    merge!(models_dict, Dict("U"=>U_model))

    # -- BUILD CNOT --
    # retrieve data from U_model
    size = U_model.global_lane_map.size;
    rx_target_lane = U_model.global_lane_map.rx_target_lane;
    ry_target_lane = U_model.global_lane_map.ry_target_lane;
    rx_cnot_model = build_CNOT(n, bit, rotation_precision, rx_target_lane);
    ry_cnot_model = build_CNOT(n, bit, rotation_precision, ry_target_lane);
    merge!(models_dict, Dict("RX_CNOT"=>rx_cnot_model))
    merge!(models_dict, Dict("RY_CNOT"=>ry_cnot_model))

    # -- BUILD TRANSITION -- 
    if bit != b
        ctrl_index = U_model.global_lane_map.cnot_param_lane
        transition_model = build_transition(bit, ctrl_index, n)
        merge!(models_dict, Dict("TRANSITION"=>transition_model))
    end

    return models_dict
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

    architecture_list = Vector{Dict{String, AbstractCustomBlock}}()

    for bit in 1:b
        # gets dictionary of the models corresponding to the specific bit
        bit_model_dict = build_model(bit, rotation_precision, training_data)

        push!(architecture_list, bit_model_dict)
    end

    # compile final architecture for visualization
    size = architecture_list[1]["U"].global_lane_map.size
    architecture = chain(size)

    for bit in 1:b
        model = architecture_list[bit]
        u_model = model["U"]
        rx_cnot_model = model["RX_CNOT"]
        ry_cnot_model = model["RY_CNOT"]
        subarchitecture = chain(size)

        # organize lanes
        rx_target_lane = u_model.global_lane_map.rx_target_lane;
        ry_target_lane = u_model.global_lane_map.ry_target_lane;

        ## get RxChain and lanes
        collected_rx_lanes = vcat([rx_target_lane], u_model.global_lane_map.rx_model_lanes, u_model.global_lane_map.rx_param_lanes);
        
        ## RyChain and lanes
        collected_ry_lanes = vcat([ry_target_lane], u_model.global_lane_map.ry_model_lanes, u_model.global_lane_map.ry_param_lanes);

        subarchitecture = chain(
            size,
            subroutine(subarchitecture, 1:size),
            subroutine(u_model.rx_compiled_architecture, collected_rx_lanes),
            subroutine(rx_cnot_model.architecture, rx_cnot_model.global_lane_map.lanes)
        )

        subarchitecture = chain(
            size,
            subroutine(subarchitecture, 1:size),
            subroutine(u_model.ry_compiled_architecture, collected_ry_lanes),
            subroutine(ry_cnot_model.architecture, ry_cnot_model.global_lane_map.lanes)
        )

        if bit != b
            subarchitecture = chain(
                size,
                subroutine(subarchitecture, 1:size),
                subroutine(model["TRANSITION"].architecture, compile_lane_map(model["TRANSITION"], b))
            )
        end

        architecture = chain(
            size,
            subroutine(architecture, 1:size),
            subroutine(subarchitecture, 1:size)
        )
    end

    return OAABlock(
        architecture,
        architecture_list,
        rotation_precision,
        b,
        n,
        size
    );
end

# compiles and runs the given circuit using OAA
# by iterating over each of the subblocks for bit-by-bit training
# e.g. training based on the given circuit
function run_oaa(skeleton::OAABlock)
    b = skeleton.num_bits # number of bits in initial training data

    # set up initial state
    state = zero_state(skeleton.total_num_lanes + b);

    # define R0lstar
    R0lstar = chain(
        skeleton.num_training_data + skeleton.rotation_precision + 1,
        repeat(X, skeleton.num_training_data + 2:skeleton.num_training_data + skeleton.rotation_precision + 1),
        cz(skeleton.num_training_data + 2:skeleton.num_training_data + skeleton.rotation_precision, skeleton.num_training_data + skeleton.rotation_precision + 1),
        repeat(X, skeleton.num_training_data + 2:skeleton.num_training_data + skeleton.rotation_precision + 1),
    );

    MAX_ITER = 3;

    # iterate over the subblocks to train bit-by-bit
    for i in 1:b
        u_model = skeleton.architecture_list[i]["U"]
        rx_cnot_model = skeleton.architecture_list[i]["RX_CNOT"]
        ry_cnot_model = skeleton.architecture_list[i]["RY_CNOT"]

        # organize lanes
        rx_target_lane = u_model.global_lane_map.rx_target_lane;
        ry_target_lane = u_model.global_lane_map.ry_target_lane;

        ## get RxChain and lanes
        collected_rx_lanes = vcat([rx_target_lane], u_model.global_lane_map.rx_model_lanes, u_model.global_lane_map.rx_param_lanes);
        
        ## RyChain and lanes
        collected_ry_lanes = vcat([ry_target_lane], u_model.global_lane_map.ry_model_lanes, u_model.global_lane_map.ry_param_lanes);
        
        # run state through first model
        for j in 1:MAX_ITER
            ## focus Rx lanes
            focus!(state, collected_rx_lanes);

            ## pipe state into RxChain
            state |> u_model.rx_compiled_architecture;
            relax!(state, collected_rx_lanes);

            focus!(state, rx_cnot_model.global_lane_map.lanes);
            state |> rx_cnot_model.architecture;
            relax!(state, rx_cnot_model.global_lane_map.lanes);

            focus!(state, collected_rx_lanes);
            ## measure outcome
            outcome = measure!(state, 1);

            ## if outcome == 0, run OAA again
            if outcome == 0
                state |> Daggered(u_model.rx_compiled_architecture);
                state |> R0lstar;
                state |> u_model.rx_compiled_architecture;
                ## relax Rx lanes
                relax!(state, collected_rx_lanes);
            else
                ## relax Rx lanes
                break
            end
        end

        relax!(state, collected_rx_lanes);
        

        ## focus Ry lanes
        for j in 1:MAX_ITER
            focus!(state, collected_ry_lanes);

            ## pipe state into RyChain
            state |> u_model.ry_compiled_architecture;
            relax!(state, collected_ry_lanes);

            focus!(state, ry_cnot_model.global_lane_map.lanes);
            state |> ry_cnot_model.architecture;
            relax!(state, ry_cnot_model.global_lane_map.lanes);

            focus!(state, collected_ry_lanes);

            ## measure outcome
            outcome = measure!(state, 1);

            ## if outcome != 0, run OAA again
            if outcome == 0
                state |> Daggered(u_model.ry_compiled_architecture);
                state |> R0lstar;
                state |> u_model.ry_compiled_architecture;
                ## relax Ry lanes
                relax!(state, collected_ry_lanes);
            else
                break
            end

        end
        
        ## relax Ry lanes
        relax!(state, collected_ry_lanes);

        # if i != b
            # append the transition model
        if i != b
            transition_model = skeleton.architecture_list[i]["TRANSITION"]
            transition_lane_map = compile_lane_map(transition_model)
            focus!(state, transition_lane_map)
            state |> transition_model.architecture
            relax!(state, transition_lane_map)
        end
    end

    return state
end