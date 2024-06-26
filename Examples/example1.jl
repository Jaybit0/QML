# ======== IMPORTS ========
# =========================

if !isdefined(Main, :QML)
    include("../Modules/QML.jl")
    using .QML
end

using Yao
using Yao.EasyBuild, YaoPlots

# ========== CODE ==========
# ==========================

# Initialize an empty circuit with 1 target lane and 3 model lanes
mcirc = chain(4, repeat(4, H, 2:4), control(2, 1=>Ry(π)), control(3, 1=>Ry(π/2)), control(4, 1=>Ry(π/4)))

model_lane = 1
param_lanes = 2:4

grover = QMLBlock(mcirc, model_lane, param_lanes, [[true], [false], [true], [true], [true]]; log=true)

#println(grover.compiled_circuit.main_circuit)
# Vizualize the main circuit
#vizcircuit(grover.compiled_circuit.main_circuit)

# Uncomment this to vizualize the measured results
#register = zero_state(Yao.nqubits(grover))
#measured = register |> grover |> r->measure(r; nshots=100000)
#plotmeasure(measured; oracle_function=grover.compiled_circuit.oracle_function, sort=true, num_entries=14)
plotmeasure(grover; sort=true, num_entries=14)