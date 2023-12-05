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

grover_circ = empty_circuit(2, 4)

hadamard(grover_circ, 3:6)
rotation(grover_circ, 1, control_lanes = 3:4; max_rotation_rad = pi)
rotation(grover_circ, 2, control_lanes = 5:6; max_rotation_rad = pi)
not(grover_circ, 1, control_lanes = [3:4])
#not(grover_circ, 1, control_lanes = 2)

#rotation(grover_circ, 1, control_lanes = 5:5, max_rotation_rad = pi)
#rotation(grover_circ, 2, control_lanes = 6:6, max_rotation_rad = pi)
#rotation(grover_circ, 3, control_lanes = 7:7, max_rotation_rad = pi)
#rotation(grover_circ, 4, control_lanes = 8:8, max_rotation_rad = pi)
#not(grover_circ, 2:4, control_lanes = 1:3)

#main_circ = compile_circuit(grover_circ)
#reg = zero_state(8)
#out = reg |> main_circ

function count(measured, p, out, criterion)
    oracle = _wrap_oracle(_oracle_function(1:2, criterion), 1:2)

    cum_prob = 0

    num_correct = 0
    num_correct_oracle = 0
    num_all = length(measured)
    m = zeros(length(p))
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
        if abs(m[i] / num_all - p[i]) > .01
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
end

criterion = [false, false]
out, main_circ, grov = auto_compute(grover_circ, 1:2, criterion)

vizcircuit(main_circ)

#p = probs(out)
#measured = out |> r->measure(r; nshots=100000)

#count(measured, p, out, criterion)
#plotmeasure(measured)