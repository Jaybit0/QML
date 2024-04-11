export GroverMLBlock

include("GroverCircuitBuilder.jl")

using .GroverML
using Yao
using Yao.EasyBuild, YaoPlots

# ========== CODE ==========
# ==========================

struct CompiledGroverCircuit
    out::Union{Yao.ArrayReg, Nothing}
    main_circuit::Yao.AbstractBlock
    grover_circuit::Yao.AbstractBlock
    chained::Yao.AbstractBlock
    oracle_function::Function
end

struct GroverMLBlock{D} <: CompositeBlock{D} 
    circuit::GroverCircuit
    output_bits::Union{Vector, Bool}
    grover_iterations::Int
    compiled_circuit::CompiledGroverCircuit
end

function GroverMLBlock(circuit::AbstractBlock, model_lanes::Union{Vector, AbstractRange, Int}, param_lanes::Union{AbstractRange, Vector, Int}, output_bits::Union{Vector, Bool}; grover_iterations::Union{Int, Nothing}=nothing, log::Bool=false, evaluate::Bool=false, start_register::Union{Yao.ArrayReg, Nothing} = nothing)
    block_size = nqubits(circuit)
    
    if isa(model_lanes, Int)
        model_lanes = [model_lanes]
    end

    if isa(param_lanes, Int)
        param_lanes = [param_lanes]
    end
    
    if length(model_lanes) + length(param_lanes) != block_size
        throw(ArgumentError("The model lanes and parameter lanes must cover all qubits in the circuit"))
    end

    mcircuit = empty_circuit(model_lanes, param_lanes)
    yao_block(mcircuit, [1:block_size], circuit)

    out, main_circuit, grover_circuit, oracle_function, actual_grover_iterations = auto_compute(mcircuit, output_bits; forced_grover_iterations=grover_iterations, evaluate=evaluate, log=log, new_mapping_system=true, evaluate_optimal_grover_n=isnothing(grover_iterations), start_register=start_register)
    new_block_size = nqubits(main_circuit)
    chained = chain(new_block_size, put(1:new_block_size => main_circuit), put(1:new_block_size => grover_circuit))
    compiled_circuit = CompiledGroverCircuit(out, main_circuit, grover_circuit, chained, oracle_function)

    return GroverMLBlock{2}(mcircuit, output_bits, actual_grover_iterations, compiled_circuit)
end

function Yao.apply!(reg::Yao.AbstractRegister, gate::GroverMLBlock)
    return Yao.apply!(reg, chained(gate))
end

function Yao.nqubits(gate::GroverMLBlock)
    return nqubits(gate.compiled_circuit.main_circuit)
end

function Yao.nqudits(gate::GroverMLBlock)
    return nqudits(gate.compiled_circuit.main_circuit)
end

function chained(gate::GroverMLBlock)::AbstractBlock
    return gate.compiled_circuit.chained
end

function Yao.subblocks(gate::GroverMLBlock)
    return subblocks(chained(gate))
end

function Yao.mat(gate::GroverMLBlock)
    return mat(chained(gate))
end

function Yao.mat(::Type{T}, gate::GroverMLBlock) where {T}
    return mat(T, chained(gate))
end

function Yao.occupied_locs(gate::GroverMLBlock)
    return occupied_locs(chained(gate))
end

function Yao.print_block(gate::GroverMLBlock)
    return print_block(chained(gate))
end

function Yao.adjoint(gate::GroverMLBlock)
    return adjoint(chained(gate))
end

function YaoPlots.draw!(grid::CircuitGrid, gate::GroverMLBlock, args...; kwargs...)
    return YaoPlots.draw!(grid, chained(gate), args...; kwargs...)
end