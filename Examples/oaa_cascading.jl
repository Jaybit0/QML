if !isdefined(Main, :QML)
    include("../Modules/QML.jl")
    using .QML
end

using Yao, YaoPlots
using Yao.EasyBuild


mutable struct RChainBlock
    model_architecture
    lanes::Vector{Int}
end


function create_RChainBlock(i, param_lanes, num_model_lanes)

    println("Creating RxChain")
    RxChain = chain(num_model_lanes + 1, repeat(H, 2:num_model_lanes + 1),
        control(2, 1=>Rx(π))
    );
    println("Creating RyChain")
    RyChain = chain(num_model_lanes + 1, repeat(H, 2:num_model_lanes + 1),
        control(2, 1=>Ry(π))
    );

    if i == size(param_lanes)[1]
        println("Creating chain without CNOT")


        arch_size = 2 * num_model_lanes + 1

        # architecture = chain(arch_size, 
        #     put(1:(1 + num_model_lanes) => RxChain),
        #     put(pushfirst!(collect(num_model_lanes + 2:2 * num_model_lanes + 1), 1) => RyChain)
        # );

        architecture = chain(arch_size, 
            subroutine(RxChain, 1:(1 + num_model_lanes)),
            subroutine(RyChain, pushfirst!(collect(num_model_lanes + 2:2 * num_model_lanes + 1), 1))
        );

        x_range = collect(
            (size(param_lanes)[1] + (2 * num_model_lanes * (i - 1)) + 1)
            :(size(param_lanes)[1] + (2 * num_model_lanes * (i - 1)) + num_model_lanes - 1)
        )
        y_range = collect(
            (size(param_lanes)[1] + (2 * num_model_lanes * (i - 1)) + num_model_lanes)
            :(size(param_lanes)[1] + 1 + 2 * num_model_lanes * i - 1)
        )

        chain_lanes = append!(
                pushfirst!(x_range, i),
                y_range
        )
    else    
        println("Creating chain with CNOT")
        arch_size = 2 * num_model_lanes + 2
        # architecture = chain(arch_size, 
        #     put(1:(1 + num_model_lanes) => RxChain),
        #     put(pushfirst!(collect(num_model_lanes + 2:2 * num_model_lanes + 1), 1) => RyChain),
        #     cnot(1, 4)
        # );

        architecture = chain(arch_size, 
            subroutine(RxChain, 1:(1 + num_model_lanes)),
            subroutine(RyChain, pushfirst!(collect(num_model_lanes + 2:2 * num_model_lanes + 1), 1)),
            cnot(1, 4)
        );

        x_range = collect(
            (size(param_lanes)[1] + (2 * num_model_lanes * (i - 1)) + 1)
            :(size(param_lanes)[1] + (2 * num_model_lanes * (i - 1)) + num_model_lanes - 1)
        )
        y_range = collect(
            (size(param_lanes)[1] + (2 * num_model_lanes * (i - 1)) + num_model_lanes)
            :(size(param_lanes)[1] + 1 + 2 * num_model_lanes * i - 1)
        )

        chain_lanes = push!(
            append!(
                pushfirst!(x_range, i), y_range),
                i + 1
        )
    end

    return RChainBlock(architecture, chain_lanes)

end

param_lanes = 1:2

num_model_lanes = 1

subchains = RChainBlock[]

for i in 1:size(param_lanes)[1]
    r = create_RChainBlock(i, param_lanes, num_model_lanes)
    push!(subchains, r)
end



# non-functioning code
# ma = chain(max_num_qubits, subroutine(r.model_architecture, r.lanes) for r in subchains)

for r in subchains
    println(r.lanes)
    println(r.model_architecture)
    println()
end

max_num_qubits = size(param_lanes)[1] + num_model_lanes * 2 * size(param_lanes)[1]

ma = chain(
    max_num_qubits,
    subroutine(r.model_architecture, r.lanes) for r in subchains
)

vizcircuit(ma)
