# ======== IMPORTS ========
# =========================

if !isdefined(Main, :GroverML)
    include("../Modules/GroverML.jl")
    using .GroverML
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
grover = GroverMLBlock(mcirc, model_lanes, param_lanes, true; log=true)

@info "===== GROVER 2 ====="
mcirc2 = chain(3, put(2:3 => grover), control(3, 1 => Ry(π/2)))
grover2 = GroverMLBlock(mcirc2, 1, 2:3, [[true], [false]]; log=true)

# Vizualize the main circuit
vizcircuit(grover2.compiled_circuit.main_circuit)

# Uncomment this to vizualize the measured results
#plotmeasure(grover2; sort=true, num_entries=14)