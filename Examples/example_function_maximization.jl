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

# Set to true to augment the circuit by rescaling with l := 0.75 and repeating measurements 3 times
augment = false


maxval = augment ? π*0.75 : π
batch = augment ? [[true], [true], [true]] : [[true]]

model_lanes = 1
param_lanes = 2:5

# Initialize the circuit
circuit = chain(5, repeat(H, param_lanes), control(2, 1 => Ry(maxval / 2)), control(3, 1 => Ry(maxval / 4)), control(4, 1 => Ry(maxval / 8)), control(5, 1 => Ry(maxval / 16)))

grover = QMLBlock(circuit, model_lanes, param_lanes, batch; log=true)

#vizcircuit(grover.compiled_circuit.main_circuit)

plotmeasure(grover; sort=true, num_entries=10)