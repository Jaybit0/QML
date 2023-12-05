# Setup virtual environment as it doesn't work for me locally otherwise
# These three lines of code must be executed before anything else, even imports
include("../Modules/SetupTool.jl")

using .SetupTool

setupPackages(false, update_registry = false)

using Revise

Revise.includet("../Modules/GroverML.jl")

using .GroverML

Revise.includet("../Modules/GroverCircuitBuilder.jl")

using .GroverCircuitBuilder

Revise.includet("../Modules/GroverPlotting.jl")

using .GroverPlotting

configureYaoPlots()

using Yao
using Yao.EasyBuild, YaoPlots

grover_circ = GroverCircuit(2::Int, 6::Int, Vector{GroverBlock}())

hadamard(grover_circ, 1:8)
rotation(grover_circ, 1, control_lanes = 3:4, max_rotation_rad = pi/2)
rotation(grover_circ, 2, control_lanes = 5:6, max_rotation_rad = pi/2)
not(grover_circ, 2, control_lanes = 1)
#rotation(grover_circ, 1, control_lanes = 5:5, max_rotation_rad = pi)
#rotation(grover_circ, 2, control_lanes = 6:6, max_rotation_rad = pi)
#rotation(grover_circ, 3, control_lanes = 7:7, max_rotation_rad = pi)
#rotation(grover_circ, 4, control_lanes = 8:8, max_rotation_rad = pi)
#not(grover_circ, 2:4, control_lanes = 1:3)

#main_circ = compile_circuit(grover_circ)
#reg = zero_state(8)
#out = reg |> main_circ

function count(measured, p, out)
    criterion = [true, false]
    oracle = _wrap_oracle(_oracle_function(1:2, criterion), 1:2)

    cum_prob = 0

    num_correct = 0
    num_correct_oracle = 0
    num_all = length(measured)
    m = zeros(2^8)
    for x in measured
        m[Int(x)+1] += 1
        if x[1] == criterion[1] && x[2] == criterion[2]
            num_correct += 1
        end

        if oracle(Int(x)+1)
            num_correct_oracle += 1
        end
    end

    diff = 0
    for i in eachindex(m)
        diff += abs(m[i] / num_all - p[i])
        if abs(m[i] / num_all - p[i]) > .001
            println(bitstring(i)[end-7:end], ": ", m[i] / num_all, "<->", p[i])
        end

        if oracle(i)
            cum_prob += p[i]
        end
    end

    println("Diff: ", diff)
    println(num_correct, "/", num_all)
    println(num_correct_oracle, "/", num_all)
    println("Cumulative Probability: ", cum_prob)
    #println("Cumulative Probability 3: ", computeCumProb(out, _wrap_oracle(_oracle_function(1:2, criterion), 1:2)))
end

out = auto_compute(grover_circ, 1:2, [true, false], forced_grover_iterations=2)

#vizcircuit(build_grover_iteration(grover_circ, 1:2, [true, false]))

p = probs(out)
println("Cumulative Probability 1: ", computeCumProb(out, _wrap_oracle(_oracle_function(1:2, [true, false]), 1:2)))
measured = out |> r->measure(r; nshots=100000)
println("Cumulative Probability 2: ", computeCumProb(out, _wrap_oracle(_oracle_function(1:2, [true, false]), 1:2)))

count(measured, p, out)
#plotmeasure(measured)