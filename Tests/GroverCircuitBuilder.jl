# ======== IMPORTS ========
# =========================

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
using Test

@testset "Helper functions" begin
    @testset "_diffusion_gate_for_zero_function" begin
        grover_circ = empty_circuit(1, 1)
        @test GroverCircuitBuilder._diffusion_gate_for_zero_function(2, 2, true) == chain(2, put(2 => Z))
        @test GroverCircuitBuilder._diffusion_gate_for_zero_function(2, 2, false) == chain(2, put(2 => X), put(2 => Z), put(2 => X))
    end
end