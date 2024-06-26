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

# mutable struct ModelBlock
#     architecture::ChainBlock
#     rx_chain::ChainBlock
#     ry_chain::ChainBlock
# end
# mutable struct RChainBlock
#     model::ModelBlock
#     lanes::Vector{Int}
# end

# mutable struct OAABlock
#     architecture::ChainBlock
#     subchains::Vector{RChainBlock}
#     num_model_lanes::Int
#     total_num_lanes::Int
#     rotation_precision::Int
# end

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
    
    # size::Int
    # lanes::Vector{Int}
    # rx_model_lanes::Vector{Int}
    # ry_model_lanes::Vector{Int}
    # target_lane::Int
    # rx_param_lanes::Vector{Int}
    # ry_param_lanes::Vector{Int}
    # cnot_param_lane::Int

    return GlobalLaneMap(
        n*(b+1)+2*rp*b + b, # size,
        1:n*(b+1)+2*rp*b + b, # lanes
        n*(bit - 1) + 1:n*(bit - 1) + n, # rp x-model
        n*(bit - 1) + 1:n*(bit - 1) + n, # rp y-model
        n*b + bit, # target bit
        n*(b + 1) + 2*rp*(bit - 1) + 1:n*(b + 1) + 2*rp*(bit - 1) + rp, # rp x-param
        n*(b + 1) + 2*rp*(bit - 1) + rp + 1:n*(b + 1) + 2*rp*(bit - 1) + 2*rp, # rp y-param
        n*(b+1)+2*rp*b + 2*rp + bit # cnot param lane
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


# function old_run_OAA(model::OAABlock, state::ArrayReg{2, ComplexF64, Matrix{ComplexF64}})
    
#     R0lstar = chain(
#         model.num_model_lanes + model.rotation_precision,
#         repeat(X, model.num_model_lanes + 1:model.num_model_lanes + rotation_precision),
#         cz(model.num_model_lanes + 1:model.num_model_lanes + rotation_precision - 1, model.num_model_lanes + rotation_precision),
#         repeat(X, model.num_model_lanes + 1:model.num_model_lanes + rotation_precision)
#     )

#     # get Rx lanes
#     rx_lanes = collect(1:model.num_model_lanes)
#     ry_lanes = collect(1:model.num_model_lanes)

#     for i in size(model.subchains)
#         # x
#         append!(rx_lanes, 
#             collect((num_model_lanes + (2 * rotation_precision * (i - 1)) + 1)
#             :(num_model_lanes + (2 * rotation_precision * (i - 1)) + rotation_precision - 1))
#         )

#         # y
#         append!(ry_lanes, 
#             collect((num_model_lanes + (2 * rotation_precision * (i - 1)) + rotation_precision)
#             :(num_model_lanes + 1 + 2 * rotation_precision * i - 1))
#         )
#     end

#     append_qubits!(state, model.num_model_lanes * model.rotation_precision)
    
#     focus!(state, rx_lanes)

#     state |> model.model_architecture.RxChain

#     outcome = measure!(state, rx_lanes)
#     if (outcome != 0)
#         state |> Daggered(model.model_architecture.RxChain);
#         state |> R0lstar;
#         state |> model.model_architecture.RxChain;
#     end

#     relax!(state, rx_lanes)

# end

# # DONE: generalize to general number of model lanes
# # TODO: convert to vector input
# # PAUSED: create mapping function 
# function old_create_RChainBlock(i, num_model_lanes, rotation_precision)

#     ctrl_rotx(ctrl, target, θ) = control(ctrl, target => Rx(θ))
#     ctrl_roty(ctrl, target, θ) = control(ctrl, target => Ry(θ))

#     # define the set of rotations to be applied in model
#     max_rotation = 2*π
#     rotation_increment = max_rotation / (rotation_precision + 1)

#     RxModel(target) = chain(rotation_precision + 1, ctrl_rotx(j + 1, target, rotation_increment * j) for j in 1:rotation_precision);
#     RyModel(target) = chain(rotation_precision + 1, ctrl_roty(j + 1, target, rotation_increment * j) for j in 1:rotation_precision);

#     RxChain = chain(
#         rotation_precision + 1,
#         repeat(H, 2:rotation_precision + 1),
#         subroutine(RxModel(1), 1:rotation_precision + 1)
#     );

#     RyChain = chain(
#         rotation_precision + 1,
#         repeat(H, 2:rotation_precision + 1),
#         subroutine(RyModel(1), 1:rotation_precision + 1)
#     );

#     x_lanes = collect(
#         (num_model_lanes + (2 * rotation_precision * (i - 1)) + 1)
#         :(num_model_lanes + (2 * rotation_precision * (i - 1)) + rotation_precision - 1)
#     )

#     y_lanes = collect(
#         (num_model_lanes + (2 * rotation_precision * (i - 1)) + rotation_precision)
#         :(num_model_lanes + 1 + 2 * rotation_precision * i - 1)
#     )

#     if i == num_model_lanes
#         arch_size = 2 * rotation_precision + 1

#         architecture = chain(arch_size, 
#             subroutine(RxChain, 1:(1 + rotation_precision)),
#             subroutine(RyChain, pushfirst!(collect(rotation_precision + 2:2 * rotation_precision + 1), 1))
#         );

        
#         chain_lanes = append!(
#                 pushfirst!(x_lanes, i),
#                 y_lanes
#         )

#     else   
#         arch_size = 2 * rotation_precision + 2

#         architecture = chain(arch_size, 
#             subroutine(RxChain, 1:(1 + rotation_precision)),
#             subroutine(RyChain, pushfirst!(collect(rotation_precision + 2:2 * rotation_precision + 1), 1)),
#             cnot(1, arch_size)
#         );

#         x_lanes = collect(
#             (num_model_lanes + (2 * rotation_precision * (i - 1)) + 1)
#             :(num_model_lanes + (2 * rotation_precision * (i - 1)) + rotation_precision - 1)
#         )
#         y_lanes = collect(
#             (num_model_lanes + (2 * rotation_precision * (i - 1)) + rotation_precision)
#             :(num_model_lanes + 1 + 2 * rotation_precision * i - 1)
#         )

#         chain_lanes = push!(
#             append!(
#                 pushfirst!(x_lanes, i), y_lanes),
#                 i + 1
#         )
#     end

#     return RChainBlock(
#         ModelBlock(architecture, RxChain, RyChain),
#         chain_lanes
#     )
# end

# # TODO: add visualization component
# # DONE: rename components
# function old_create_OAACircuit(num_model_lanes, rotation_precision)
#     subchains = RChainBlock[]

#     for i in 1:num_model_lanes
#         r = create_RChainBlock(i, num_model_lanes, rotation_precision)
#         push!(subchains, r)
#     end

#     max_num_qubits = num_model_lanes + rotation_precision * 2 * num_model_lanes

#     ma = chain(
#         max_num_qubits,
#         subroutine(r.model.architecture, r.lanes) for r in subchains
#     )
#     return OAABlock(
#         ma,
#         subchains,
#         num_model_lanes,
#         num_model_lanes * (rotation_precision + 1) * 2,
#         rotation_precision
#     )
# end


# mutable struct RSubchainBlock
#     architecture::ChainBlock
#     lanes::Vector{Int}
# end

# # DONE: create run function for OAA algorithm 
# # TODO: implement checks to make sure number qubits state and model lanes match
# function run_OAA(model::OAABlock, state::ArrayReg{2, ComplexF64, Matrix{ComplexF64}})
#     R0lstar = chain(
#         model.num_model_lanes * (1 + model.rotation_precision),
#         repeat(X, model.num_model_lanes + 1:model.num_model_lanes + rotation_precision),
#         cz(model.num_model_lanes + 1:model.num_model_lanes + rotation_precision - 1, model.num_model_lanes + rotation_precision),
#         repeat(X, model.num_model_lanes + 1:model.num_model_lanes + rotation_precision)
#     );

#     append_qubits!(state, num_model_lanes * rotation_precision * 2)

#     focus!(state, model.rx_model.lanes)

#     state |> model.rx_model.architecture;

#     outcome = measure!(state, num_model_lanes * (1 + rotation_precision))
#     if (outcome != 0)
#         state |> Daggered(model.rx_model.architecture);
#         state |> R0lstar;
#         state |> model.rx_model.architecture;
#     end
#     relax!(state, model.rx_model.lanes)

#     focus!(state, model.ry_model.lanes)

#     state |> model.ry_model.architecture;
    
#     outcome = measure!(state, 1:num_model_lanes * (1 + rotation_precision))

#     if (outcome != 0)
#         state |> Daggered(model.ry_model.architecture);
#         state |> R0lstar;
#         state |> model.ry_model.architecture;
#     end
#     relax!(state, model.ry_model.lanes)
# end

# function create_rx_model(num_model_lanes, rotation_precision)
#     # create template for model with specified rotation precision
#     ctrl_rotx(ctrl, target, θ) = control(ctrl, target => Rx(θ));
#     rotation_increment = MAX_ROTATION / (rotation_precision + 1);
    
#     RxModel = chain(rotation_precision + 1, ctrl_rotx(j + 1, 1, rotation_increment * j) for j in 1:rotation_precision);

#     RxChain = chain(
#         rotation_precision + 1,
#         repeat(H, 2:rotation_precision + 1),
#         subroutine(RxModel, 1:rotation_precision + 1)
#     );

#     RxSubchainCNOT = chain(
#         rotation_precision + 2,
#         subroutine(RxChain, 1:rotation_precision + 1),
#         cnot(1, rotation_precision + 2)
#     );

#     x_lanes = collect(1:num_model_lanes)
#     append!(
#         x_lanes,
#         collect(num_model_lanes + 1:num_model_lanes + num_model_lanes * rotation_precision)
#     )
    
#     shift = 0

#     subchains = RSubchainBlock[]

#     for i in 1:num_model_lanes
#         if i != num_model_lanes
#             lanes = push!(pushfirst!(
#                     collect(shift + num_model_lanes + (i - 1) * rotation_precision + 1:shift + num_model_lanes + i * rotation_precision),
#                     i
#                 ), i + 1)
#             architecture = RxSubchainCNOT
#         else
#             lanes = pushfirst!(
#                         collect(shift + num_model_lanes + (i - 1) * rotation_precision + 1:shift + num_model_lanes + i * rotation_precision),
#                         i
#             )
#             architecture = RxChain
#         end
#         push!(subchains, RSubchainBlock(architecture, lanes))
#     end

#     rx_model = chain(
#         num_model_lanes * (1 + rotation_precision),
#         subroutine(r.architecture, r.lanes) for r in subchains
#     )

#     return RChainBlock(rx_model, x_lanes)
# end

# function create_ry_model(num_model_lanes, rotation_precision)
#     # create template for model with specified rotation precision
#     ctrl_roty(ctrl, target, θ) = control(ctrl, target => Ry(θ));
#     rotation_increment = MAX_ROTATION / (rotation_precision + 1);
#     RyModel = chain(rotation_precision + 1, ctrl_roty(j + 1, 1, rotation_increment * j) for j in 1:rotation_precision);

#     RyChain(i) = chain(
#         rotation_precision + 1,
#         repeat(H, 2:rotation_precision + 1),
#         subroutine(RyModel, 1:rotation_precision + 1)
#     );

#     RySubchainCNOT = chain(
#         rotation_precision + 2,
#         subroutine(RyChain, 1:rotation_precision + 1),
#         cnot(1, rotation_precision + 2)
#     );

#     RySubchainEND = chain(
#         rotation_precision + 1,
#         subroutine(RyChain, 1:rotation_precision + 1)
#     );

#     y_lanes = collect(1:num_model_lanes)
#     append!(
#         y_lanes,
#         collect(num_model_lanes + num_model_lanes * rotation_precision + 1:num_model_lanes + 2 * num_model_lanes * rotation_precision)
#     )
    
#     shift = 0

#     subchains = RSubchainBlock[]

#     for i in 1:num_model_lanes
#         if i != num_model_lanes
#             lanes = push!(pushfirst!(
#                     collect(shift + num_model_lanes + (i - 1) * rotation_precision + 1:shift + num_model_lanes + i * rotation_precision),
#                     i
#                 ), i + 1)
#             architecture = RySubchainCNOT
#         else
#             lanes = pushfirst!(
#                         collect(shift + num_model_lanes + (i - 1) * rotation_precision + 1:shift + num_model_lanes + i * rotation_precision),
#                         i
#             )
#             architecture = RySubchainEND
#         end
#         push!(subchains, RSubchainBlock(architecture, lanes))
#     end

#     ry_model = chain(
#         num_model_lanes * (1 + rotation_precision),
#         subroutine(r.architecture, r.lanes) for r in subchains
#     )


#     return RChainBlock(ry_model, y_lanes)
# end

# # TODO: add visualization component
# # DONE: rename components
# function old_create_OAACircuit(num_model_lanes, rotation_precision)
#     rx_model = create_rx_model(num_model_lanes, rotation_precision)
#     ry_model = create_ry_model(num_model_lanes, rotation_precision)
    
#     total_num_lanes = num_model_lanes * (rotation_precision * 2 + 1)

#     architecture = chain(
#         total_num_lanes,
#         subroutine(rx_model.architecture, rx_model.lanes),
#         subroutine(ry_model.architecture, ry_model.lanes)
#     )

#     return OAABlock(
#         architecture,
#         rx_model,
#         ry_model,
#         num_model_lanes,
#         total_num_lanes,
#         rotation_precision
#     )
# end
