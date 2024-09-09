if !isdefined(Main, :QML)
    include("../Modules/QML.jl")
    using .QML
end
include("../Modules/OAACircuitBuilder.jl")
include("../Modules/OAAPlotting.jl")

export plotmeasure;
export get_distribution;

using StatsBase: Histogram, fit
using Plots: bar, scatter!, gr; gr()
using BitBasis
using YaoPlots


sample = [0, 0, 0, 1] 
 # for every bit in the data point
# create an empty chain with number of bits
circuit = chain(b)
for j in 1:b
    for r in 1:rotation_precision
        # x param index = 2 * rotation_precision * (j - 1) + 2*(i - 1) + r
        # y param index = 2 * rotation_precision * (j - 1) + 2*(i - 1) + rotation_precision + r
        # angle = MAX_ROTATION / 2^(r)
        # apply X rotation
        if sample[2 * rotation_precision * (j - 1) + r] == 1
            # add Rx gate with rotatin MAX_ROTATION / 2^(r)
            circuit = chain(b,
                subroutine(circuit, 1:b), # append to circuit
                put(j => Rx(MAX_ROTATION / 2^(r))) # apply Rx rotation to jth bit
            );
        end
        if sample[2 * rotation_precision * (j - 1) + rotation_precision + r] == 1
        # add Ry gate with rotation MAX_ROTATION / 2^(r)
            circuit = chain(b,
                subroutine(circuit, 1:b), # append to circuit
                put(j => Ry(MAX_ROTATION / 2^(r))) # apply Rx rotation to jth bit
            );
        end
    end

    # apply CNOT gate if parameter bit == 1
    if j != b
        if sample[length(sample)] == 1
            circuit = chain(b, 
                subroutine(circuit, 1:b),
                cnot(j, j + 1)
            )
        end
    end
    
        
    if j != b
        # add CNOT gate
        circuit = chain(b, 
            subroutine(circuit, 1:b),
            cnot(j, j+1)
        );
    end
end

vizcircuit(circuit)