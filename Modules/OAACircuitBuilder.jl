using Yao

export RChainBlock;
export OAABlock;
export MAX_ROTATION;

export create_rx_model;
export create_ry_model;
export create_OAACircuit;

# mutable struct ModelBlock
#     compiled_architecture::ChainBlock
#     RxChain::ChainBlock
#     RyChain::ChainBlock
# end

mutable struct RSubchainBlock
    architecture::ChainBlock
    lanes::Vector{Int}
end
mutable struct RChainBlock
    architecture::ChainBlock
    lanes::Vector{Int}
end
mutable struct OAABlock
    architecture::ChainBlock
    rx_model::RChainBlock
    ry_model::RChainBlock
    num_model_lanes::Int
    total_num_lanes::Int
    rotation_precision::Int
end

const MAX_ROTATION = 2*π

# TODO: create run function for OAA algorithm 
# TODO: implement checks to make sure number qubits state and model lanes match
function run_OAA(model::OAABlock, state::ArrayReg{2, ComplexF64, Matrix{ComplexF64}})
    R0lstar = chain(
        model.num_model_lanes * (1 + model.rotation_precision),
        repeat(X, model.num_model_lanes + 1:model.num_model_lanes + rotation_precision),
        cz(model.num_model_lanes + 1:model.num_model_lanes + rotation_precision - 1, model.num_model_lanes + rotation_precision),
        repeat(X, model.num_model_lanes + 1:model.num_model_lanes + rotation_precision)
    );


    append_qubits!(state, num_model_lanes * rotation_precision * 2)

    focus!(state, model.rx_model.lanes)

    state |> model.rx_model.architecture;

    outcome = measure!(state, num_model_lanes * (1 + rotation_precision))
    if (outcome != 0)
        state |> Daggered(model.rx_model.architecture);
        state |> R0lstar;
        state |> model.rx_model.architecture;
    end
    relax!(state, model.rx_model.lanes)

    focus!(state, model.ry_model.lanes)

    state |> model.ry_model.architecture;
    println(model.ry_model.lanes)
    
    outcome = measure!(state, 1:num_model_lanes * (1 + rotation_precision))

    if (outcome != 0)
        state |> Daggered(model.ry_model.architecture);
        state |> R0lstar;
        state |> model.ry_model.architecture;
    end
    relax!(state, model.ry_model.lanes)
end
# function run_OAA(model::OAABlock, state::ArrayReg{2, ComplexF64, Matrix{ComplexF64}})
    
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

function create_rx_model(num_model_lanes, rotation_precision)
    # create template for model with specified rotation precision
    ctrl_rotx(ctrl, target, θ) = control(ctrl, target => Rx(θ));
    rotation_increment = MAX_ROTATION / (rotation_precision + 1);
    
    RxModel = chain(rotation_precision + 1, ctrl_rotx(j + 1, 1, rotation_increment * j) for j in 1:rotation_precision);

    RxChain = chain(
        rotation_precision + 1,
        repeat(H, 2:rotation_precision + 1),
        subroutine(RxModel, 1:rotation_precision + 1)
    );

    RxSubchainCNOT = chain(
        rotation_precision + 2,
        subroutine(RxChain, 1:rotation_precision + 1),
        cnot(1, rotation_precision + 2)
    );

    x_lanes = collect(1:num_model_lanes)
    append!(
        x_lanes,
        collect(num_model_lanes + 1:num_model_lanes + num_model_lanes * rotation_precision)
    )
    
    shift = 0

    subchains = RSubchainBlock[]

    for i in 1:num_model_lanes
        if i != num_model_lanes
            lanes = push!(pushfirst!(
                    collect(shift + num_model_lanes + (i - 1) * rotation_precision + 1:shift + num_model_lanes + i * rotation_precision),
                    i
                ), i + 1)
            architecture = RxSubchainCNOT
        else
            lanes = pushfirst!(
                        collect(shift + num_model_lanes + (i - 1) * rotation_precision + 1:shift + num_model_lanes + i * rotation_precision),
                        i
            )
            architecture = RxChain
        end
        push!(subchains, RSubchainBlock(architecture, lanes))
    end

    rx_model = chain(
        num_model_lanes * (1 + rotation_precision),
        subroutine(r.architecture, r.lanes) for r in subchains
    )

    return RChainBlock(rx_model, x_lanes)
end

function create_ry_model(num_model_lanes, rotation_precision)
    # create template for model with specified rotation precision
    ctrl_roty(ctrl, target, θ) = control(ctrl, target => Ry(θ));
    rotation_increment = MAX_ROTATION / (rotation_precision + 1);
    RyModel = chain(rotation_precision + 1, ctrl_roty(j + 1, 1, rotation_increment * j) for j in 1:rotation_precision);

    RyChain(i) = chain(
        rotation_precision + 1,
        repeat(H, 2:rotation_precision + 1),
        subroutine(RyModel, 1:rotation_precision + 1)
    );

    RySubchainCNOT = chain(
        rotation_precision + 2,
        subroutine(RyChain, 1:rotation_precision + 1),
        cnot(1, rotation_precision + 2)
    );

    RySubchainEND = chain(
        rotation_precision + 1,
        subroutine(RyChain, 1:rotation_precision + 1)
    );

    y_lanes = collect(1:num_model_lanes)
    append!(
        y_lanes,
        collect(num_model_lanes + num_model_lanes * rotation_precision + 1:num_model_lanes + 2 * num_model_lanes * rotation_precision)
    )
    
    shift = 0

    subchains = RSubchainBlock[]

    for i in 1:num_model_lanes
        if i != num_model_lanes
            lanes = push!(pushfirst!(
                    collect(shift + num_model_lanes + (i - 1) * rotation_precision + 1:shift + num_model_lanes + i * rotation_precision),
                    i
                ), i + 1)
            architecture = RySubchainCNOT
        else
            lanes = pushfirst!(
                        collect(shift + num_model_lanes + (i - 1) * rotation_precision + 1:shift + num_model_lanes + i * rotation_precision),
                        i
            )
            architecture = RySubchainEND
        end
        push!(subchains, RSubchainBlock(architecture, lanes))
    end

    ry_model = chain(
        num_model_lanes * (1 + rotation_precision),
        subroutine(r.architecture, r.lanes) for r in subchains
    )


    return RChainBlock(ry_model, y_lanes)
end

# TODO: add visualization component
# DONE: rename components
function create_OAACircuit(num_model_lanes, rotation_precision)
    rx_model = create_rx_model(num_model_lanes, rotation_precision)
    ry_model = create_ry_model(num_model_lanes, rotation_precision)
    
    total_num_lanes = num_model_lanes * (rotation_precision * 2 + 1)

    architecture = chain(
        total_num_lanes,
        subroutine(rx_model.architecture, rx_model.lanes),
        subroutine(ry_model.architecture, ry_model.lanes)
    )

    return OAABlock(
        architecture,
        rx_model,
        ry_model,
        num_model_lanes,
        total_num_lanes,
        rotation_precision
    )
end

# # DONE: generalize to general number of model lanes
# # TODO: convert to vector input
# # PAUSED: create mapping function 
# function create_RChainBlock(i, num_model_lanes, rotation_precision)

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

#         x_range = collect(
#             (num_model_lanes + (2 * rotation_precision * (i - 1)) + 1)
#             :(num_model_lanes + (2 * rotation_precision * (i - 1)) + rotation_precision - 1)
#         )
#         y_range = collect(
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

# TODO: add visualization component
# DONE: rename components
# function create_OAACircuit(num_model_lanes, rotation_precision)
#     
    
#     subchains = RChainBlock[]

#     for i in 1:num_model_lanes
#         r = create_RChainBlock(i, num_model_lanes, rotation_precision)
#         push!(subchains, r)
#     end

#     max_num_qubits = num_model_lanes + rotation_precision * 2 * num_model_lanes

#     ma = chain(
#         max_num_qubits,
#         subroutine(r.rchain_architecture.compiled_architecture, r.lanes) for r in subchains
#     )
#     return OAABlock(
#         ma,
#         subchains,
#         num_model_lanes,
#         num_model_lanes * (rotation_precision + 1) * 2,
#         rotation_precision
#     )
# end