# ======== IMPORTS ========
# =========================

#include("../Modules/SetupTool.jl")

#using .SetupTool
#if setupPackages(false, update_registry = false)
include("../Modules/GroverML.jl")
include("../Modules/GroverCircuitBuilder.jl")
include("../Modules/GroverPlotting.jl")
#end

using .GroverML
using .GroverCircuitBuilder
using .GroverPlotting
using Yao
using Yao.EasyBuild, YaoPlots

# ========== CODE ==========
# ==========================

struct CompiledGroverCircuit
    out::Union{Yao.ArrayReg, Nothing}
    main_circuit::Yao.AbstractBlock
    grover_circuit::Yao.AbstractBlock
    oracle_function::Function
end

struct GroverMLBlock{D} <: AbstractBlock{D} 
    circuit::GroverCircuit
    output_bits::Union{Vector, Bool}
    grover_iterations::Int
    compiled_circuit::CompiledGroverCircuit
end

function GroverMLBlock(circuit::AbstractBlock, model_lanes::Union{Int, Vector{Int}, AbstractRange}, param_lanes::Union{AbstractRange, Vector, Int}, output_bits::Union{Vector, Bool}, grover_iterations::Int)
    block_size = nqubits(circuit)
    
    if length(model_lanes) + length(param_lanes) != block_size
        throw(ArgumentError("The model lanes and parameter lanes must cover all qubits in the circuit"))
    end

    mcircuit = empty_circuit(model_lanes, param_lanes)
    yao_block(mcircuit, [1:block_size], circuit)

    out, main_circuit, grover_circuit, oracle_function = auto_compute(mcircuit, output_bits; forced_grover_iterations=grover_iterations, evaluate=false, log=false)
    compiled_circuit = CompiledGroverCircuit(out, main_circuit, grover_circuit, oracle_function)

    return GroverMLBlock{nqubits(compiled_circuit.main_circuit)}(mcircuit, output_bits, grover_iterations, compiled_circuit)
end

function Yao.apply!(reg::Yao.AbstractRegister, gate::GroverMLBlock)
    n = nqubits(gate.compiled_circuit.main_circuit)
    return Yao.apply!(reg, chain(n, put(1:n => gate.compiled_circuit.main_circuit), put(1:n => gate.compiled_circuit.grover_circuit)))
end

function Yao.nqubits(gate::GroverMLBlock)
    return nqubits(gate.compiled_circuit.main_circuit)
end

function Yao.nqudits(gate::GroverMLBlock)
    return nqudits(gate.compiled_circuit.main_circuit)
end

function Yao.subblocks(gate::GroverMLBlock)
    return subblocks(gate.compiled_circuit.main_circuit)
end

function chained(gate::GroverMLBlock)
    block_size = nqubits(gate.compiled_circuit.main_circuit)

    return chain(block_size, put(1:block_size => gate.compiled_circuit.main_circuit), put(1:block_size => gate.compiled_circuit.grover_circuit))
end

function YaoPlots.draw!(grid::CircuitGrid, gate::GroverMLBlock, args...; kwargs...)
    return YaoPlots.draw!(grid, chained(gate), args...; kwargs...)
end

circ = chain(2, put(2 => H), put(1 => X))
ml_block = GroverMLBlock(circ, 1, 1, true, 1)

vizcircuit(ml_block)