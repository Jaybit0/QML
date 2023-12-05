# Setup virtual environment as it doesn't work for me locally otherwise
# These three lines of code must be executed before anything else, even imports
include("../Modules/SetupTool.jl")

using .SetupTool

setupPackages(false)

include("../Modules/GroverML.jl")

using .GroverML

include("../Modules/GroverPlotting.jl")

using .GroverPlotting

configureYaoPlots()

using Yao
using Yao.EasyBuild, YaoPlots

function prepareSimpleMLCircuit()
    return chain(5, 
		repeat(H, [2:5]), 
		control(2, 1=>Ry(pi/2)), 
		control(3, 1=>Ry(pi/2)), 
		control(4, 1=>Ry(pi/2)), 
		control(5, 1=>Ry(pi/2)))
end

function prepareSimpleMLCircuitInv()
    return chain(5, 
		control(5, 1 => Ry(-pi/2)), 
		control(4, 1 => Ry(-pi/2)), 
		control(3, 1 => Ry(-pi/2)), 
		control(2, 1 => Ry(-pi/2)), 
		repeat(H, [2:5]))
end

function simpleMLCircuitOracle(x)
	return x%2 == 0
end

function prepareDiffusionGate(j)
    return chain(j, put(1=>Z))
end

j = 5
num_grover_iterations = 2

mainCircuit = prepareSimpleMLCircuit()
diffGate = prepareDiffusionGate(j)
oracle = prepareOracleGate(j, prepareSimpleMLCircuit(), prepareSimpleMLCircuitInv())
groverIteration = createGroverIteration(j, diffGate, oracle)

groverCircuit = createGroverCircuit(j, num_grover_iterations, groverIteration)

register = ArrayReg(bit"00000")
out = register |> mainCircuit

cumPreProb = computeCumProb(out, simpleMLCircuitOracle)

println("Cumulative Pre-Probability: ", cumPreProb)
println("Angle towards orthogonal state: ", computeAngle(cumPreProb))
println("Angle towards orthogonal state (deg): ", computeAngle(cumPreProb) / pi * 180)
println("Optimal number of Grover iterations: ", computeOptimalGroverN(cumPreProb))

out = out |> groverCircuit
println("Cumulative Probability (after " * string(num_grover_iterations) * "x Grover): ", computeCumProb(out, simpleMLCircuitOracle))

vizcircuit(mainCircuit)
#measured = out |> r->measure(r, nshots=10000)
#plotmeasure(measured)
#vizcircuit(chain(j, put(1:j => mainCircuit), put(1:j => groverCircuit)))