using Yao

export RChainBlock;
export ModelLanes;

export create_RChainBlock;
export create_OAACircuit;

mutable struct RChainBlock
    model_architecture
    lanes::Vector{Int}
end

abstract type ModelLanes end


# TODO: generalize to general number of model lanes
# TODO: convert to vector input
# TODO: create mapping function
# TODO: create run function for OAA algorithm 
function create_RChainBlock(i, model_lanes, rotation_precision)

    ctrl_rotx(ctrl, target, θ) = control(ctrl, target => Rx(θ))
    ctrl_roty(ctrl, target, θ) = control(ctrl, target => Ry(θ))

    # define the set of rotations to be applied in model
    max_rotation = 2*π
    rotation_increment = max_rotation / (rotation_precision + 1)

    println(rotation_increment)

    RxModel(target) = chain(rotation_precision + 1, ctrl_rotx(j + 1, target, rotation_increment * j) for j in 1:rotation_precision);
    RyModel(target) = chain(rotation_precision + 1, ctrl_roty(j + 1, target, rotation_increment * j) for j in 1:rotation_precision);

    RxChain = chain(
        rotation_precision + 1,
        repeat(H, 2:rotation_precision + 1),
        subroutine(RxModel(1), 1:rotation_precision + 1)
    );

    RyChain = chain(
        rotation_precision + 1,
        repeat(H, 2:rotation_precision + 1),
        subroutine(RyModel(1), 1:rotation_precision + 1)
    );

    num_model_lanes = size(model_lanes)[1]

    # lane_map = map_lanes(i, num_model_lanes, rotation_precision)

    x_lanes = collect(
        (num_model_lanes + (2 * rotation_precision * (i - 1)) + 1)
        :(num_model_lanes + (2 * rotation_precision * (i - 1)) + rotation_precision - 1)
    )

    y_lanes = collect(
        (num_model_lanes + (2 * rotation_precision * (i - 1)) + rotation_precision)
        :(num_model_lanes + 1 + 2 * rotation_precision * i - 1)
    )

    if i == size(model_lanes)[1]
        arch_size = 2 * rotation_precision + 1

        architecture = chain(arch_size, 
            subroutine(RxChain, 1:(1 + rotation_precision)),
            subroutine(RyChain, pushfirst!(collect(rotation_precision + 2:2 * rotation_precision + 1), 1))
        );

        
        chain_lanes = append!(
                pushfirst!(x_lanes, i),
                y_lanes
        )

    else   
        arch_size = 2 * rotation_precision + 2

        architecture = chain(arch_size, 
            subroutine(RxChain, 1:(1 + rotation_precision)),
            subroutine(RyChain, pushfirst!(collect(rotation_precision + 2:2 * rotation_precision + 1), 1)),
            cnot(1, arch_size)
        );

        x_range = collect(
            (size(model_lanes)[1] + (2 * rotation_precision * (i - 1)) + 1)
            :(size(model_lanes)[1] + (2 * rotation_precision * (i - 1)) + rotation_precision - 1)
        )
        y_range = collect(
            (size(model_lanes)[1] + (2 * rotation_precision * (i - 1)) + rotation_precision)
            :(size(model_lanes)[1] + 1 + 2 * rotation_precision * i - 1)
        )

        chain_lanes = push!(
            append!(
                pushfirst!(x_lanes, i), y_lanes),
                i + 1
        )
    end

    return RChainBlock(architecture, chain_lanes)

end

# TODO: add visualization component
# DONE: rename components
function create_OAACircuit(model_lanes, rotation_precision)
    subchains = RChainBlock[]

    for i in 1:size(model_lanes)[1]
        r = create_RChainBlock(i, model_lanes, rotation_precision)
        push!(subchains, r)
    end

    max_num_qubits = size(model_lanes)[1] + rotation_precision * 2 * size(model_lanes)[1]

    ma = chain(
        max_num_qubits,
        subroutine(r.model_architecture, r.lanes) for r in subchains
    )
    return ma
end