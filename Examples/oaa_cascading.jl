if !isdefined(Main, :QML)
    include("../Modules/QML.jl")
    using .QML
end

using Yao, YaoPlots


mutable struct RChainBlock
    model_architecture
    lanes::Vector{Int}
end

# TODO: insert checks for input
function create_RChainBlock(i, model_lanes, rotation_precision)

    println("Creating RxChain")
    RxChain = chain(rotation_precision + 1, repeat(H, 2:rotation_precision + 1),
        control(2, 1=>Rx(π))
    );
    println("Creating RyChain")
    RyChain = chain(rotation_precision + 1, repeat(H, 2:rotation_precision + 1),
        control(2, 1=>Ry(π))
    );

    if i == size(model_lanes)[1]
        println("Creating chain without CNOT")


        arch_size = 2 * rotation_precision + 1

        architecture = chain(arch_size, 
            subroutine(RxChain, 1:(1 + rotation_precision)),
            subroutine(RyChain, pushfirst!(collect(rotation_precision + 2:2 * rotation_precision + 1), 1))
        );

        x_range = collect(
            (size(model_lanes)[1] + (2 * rotation_precision * (i - 1)) + 1)
            :(size(model_lanes)[1] + (2 * rotation_precision * (i - 1)) + rotation_precision - 1)
        )
        y_range = collect(
            (size(model_lanes)[1] + (2 * rotation_precision * (i - 1)) + rotation_precision)
            :(size(model_lanes)[1] + 1 + 2 * rotation_precision * i - 1)
        )

        chain_lanes = append!(
                pushfirst!(x_range, i),
                y_range
        )
    else    
        println("Creating chain with CNOT")
        arch_size = 2 * rotation_precision + 2

        architecture = chain(arch_size, 
            subroutine(RxChain, 1:(1 + rotation_precision)),
            subroutine(RyChain, pushfirst!(collect(rotation_precision + 2:2 * rotation_precision + 1), 1)),
            cnot(1, 4)
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
                pushfirst!(x_range, i), y_range),
                i + 1
        )
    end

    return RChainBlock(architecture, chain_lanes)

end

model_lanes = 1:3

# TODO: generalize to general number of model lanes
# TODO: convert to vector input
rotation_precision = 1

subchains = RChainBlock[]

for i in 1:size(model_lanes)[1]
    r = create_RChainBlock(i, model_lanes, rotation_precision)
    push!(subchains, r)
end



# non-functioning code
# ma = chain(max_num_qubits, subroutine(r.model_architecture, r.lanes) for r in subchains)

for r in subchains
    println(r.lanes)
    println(r.model_architecture)
    println()
end

max_num_qubits = size(model_lanes)[1] + rotation_precision * 2 * size(model_lanes)[1]

ma = chain(
    max_num_qubits,
    subroutine(r.model_architecture, r.lanes) for r in subchains
)

vizcircuit(ma)
