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

# Initialize an empty circuit with 2 target lane and 3 model lanes
mcirc = chain(2, put(2 => H), control(2, 1 => Ry(π/4)))

model_lanes = 1
param_lanes = 2

@info "===== GROVER 1 ====="
grover = QMLBlock(mcirc, model_lanes, param_lanes, true; log=true)

reg = zero_state(5) |> chain(5, put(4:5 => grover))

@info "===== GROVER 2 ====="
mcirc2 = chain(3, control(3, 1 => Ry(π/2)))
grover2 = QMLBlock(mcirc2, 1, 2:3, [[true], [false]]; log=true, start_register=reg)

total_circ = chain(5, put(4:5 => grover), put(1:5 => grover2))

# Vizualize the main circuit
#vizcircuit(total_circ)

# Uncomment this to vizualize the measured results
register = zero_state(Yao.nqubits(total_circ))
measured = register |> total_circ |> r->measure(r; nshots=100000)
plotmeasure(measured; oracle_function=grover2.compiled_circuit.oracle_function, sort=true, num_entries=14)