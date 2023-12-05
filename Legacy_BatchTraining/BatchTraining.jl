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

# ===== ACTUAL CODE STARTS HERE =====
# ===================================

using Yao
using Yao.EasyBuild, YaoPlots

function hadamard(circuitSize::Int; range::AbstractRange = 1:circuitSize)
    return chain(circuitSize, repeat(H, range))
end

function simpleRotationGate(circuitSize::Int; granularity::Int = 4, maxRotationRad::Number = 2*pi, evalGate::Int = 1, controlRange::AbstractRange = 2:granularity+1)::Yao.YaoAPI.AbstractBlock
    granularity = length(controlRange)
    angle = maxRotationRad / granularity
    circuit = chain(circuitSize)

    for i in controlRange
        circuit = chain(circuitSize, put(1:circuitSize => circuit), control(i, evalGate => Ry(angle)))
    end

    return circuit
end

function simpleRotationGateInv(circuitSize::Int; granularity::Int = 4, maxRotationRad::Number = 2*pi, evalGate::Int = 1, controlRange::AbstractRange = 2:granularity+1)::Yao.YaoAPI.AbstractBlock
    granularity = length(controlRange)
    angle = maxRotationRad / granularity
    circuit = chain(circuitSize)

    for i in reverse(controlRange)
        circuit = chain(circuitSize, put(1:circuitSize => circuit), control(i, evalGate => Ry(-angle)))
    end

    return circuit
end

function diffusionGateForZeroFunction(circuitSize::Int, outputRange::AbstractRange, targetData::Vector{Bool})::Yao.YaoAPI.AbstractBlock
    if length(outputRange) != length(targetData)
        throw(DomainError(targetData, "Target data and output range has to be of same length!"))
    end

    circuit = chain(circuitSize)

    c = 0
    for i in outputRange
        if targetData[i]
            circuit = chain(circuitSize, put(1:circuitSize => circuit), put(i => Z))
        else
            circuit = chain(circuitSize, put(1:circuitSize => circuit), put(i => X), put(i => Z), put(i => X))
        end
        c += 1
    end

    return circuit
end

function convertToBools(index::Int, outRange::AbstractRange)::Vector{Bool}
    out = falses(length(outRange))

    c = 1
    for i in outRange
        out[c] = ((index - 1) >> (i - 1)) & 1
        c+=1
    end
    return out
end

function wrapOracle(oracle::Function, outRange::AbstractRange)
    return idx -> oracle(convertToBools(idx, outRange))
end

cSize = 4

rot1 = simpleRotationGate(cSize, evalGate=1, controlRange=3:4, maxRotationRad=pi/2)
rot2 = simpleRotationGate(cSize, evalGate=2, controlRange=3:4, maxRotationRad=pi/2)
rot2Inv = simpleRotationGateInv(cSize, evalGate=2, controlRange=3:4, maxRotationRad=pi/2)
rot1Inv = simpleRotationGateInv(cSize, evalGate=1, controlRange=3:4, maxRotationRad=pi/2)

mainCircuit = chain(cSize, put(1:cSize => hadamard(cSize)), put(1:cSize => rot1), put(1:cSize => rot2), control(1, 2 => X))
mainCircuitInv = chain(cSize, control(1, 2 => X), put(1:cSize => rot2Inv), put(1:cSize => rot1Inv), put(1:cSize => hadamard(cSize)))

diffGate = diffusionGateForZeroFunction(cSize, 1:2, [true, false])
oracle = prepareOracleGate(cSize, mainCircuit, mainCircuitInv)

groverIteration = createGroverIteration(cSize, diffGate, oracle)



register = ArrayReg(bit"0000")
out = register |> mainCircuit

function mor(dat)
    return dat[1] && !dat[2]
end

mOracle = wrapOracle(mor, 1:2)

cumPreProb = computeCumProb(out, mOracle)

num_grover_iterations = computeOptimalGroverN(cumPreProb)
println("Cumulative Pre-Probability: ", cumPreProb)
println("Angle towards orthogonal state: ", computeAngle(cumPreProb))
println("Angle towards orthogonal state (deg): ", computeAngle(cumPreProb) / pi * 180)
println("Optimal number of Grover iterations: ", computeOptimalGroverN(cumPreProb))
println("Actual optimum: ", 1/2 * (pi/(2 * computeAngle(cumPreProb)) - 1))

# Something about the optimal grover iterations is off (3, 6, (7), 10, 13 is best not 2)
#num_grover_iterations = 6
groverCircuit = createGroverCircuit(cSize, num_grover_iterations, groverIteration)

out = out |> groverCircuit
println("Cumulative Probability (after " * string(num_grover_iterations) * "x Grover): ", computeCumProb(out, mOracle))

measured = out |> r->measure(r, nshots=10000)
#plotmeasure(measured)
vizcircuit(mainCircuit)