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

@testset "Main functions" begin
    circ = empty_circuit(2, 2)
    @test compile_circuit(circ; inv = false) == chain(4)
    @test compile_circuit(circ; inv = true) == chain(4)
    @test circuit_size(circ) == 4
    @test model_lanes(circ) == [1, 2]
    @test param_lanes(circ) == [3, 4]
    @test begin
        mcirc = create_checkpoint(circ)
        insert_model_lane(mcirc, 1)
        model_lanes(mcirc) == [2, 3, 1]
    end
    @test begin
        mcirc = create_checkpoint(circ)
        insert_param_lane(mcirc, 1)
        param_lanes(mcirc) == [4, 5, 1]
    end
    @test begin
        mcirc = create_checkpoint(circ)
        manipulate_lanes(mcirc, x -> (x % 4) + 1)
        model_lanes(mcirc) == [2, 3] && param_lanes(mcirc) == [4, 1]
    end
end

@testset "Grover blocks" begin
    @testset "Yao block" begin
        @test begin
            circ = empty_circuit(2, 4)
            custom_block = chain(2, put(1 => Rz(pi)), put(2 => Rz(pi)))
            custom_block_inv = chain(2, put(2 => Rz(-pi)), put(1 => Rz(-pi)))
            yao_block(circ, [1:2], custom_block, custom_block_inv; control_lanes=[3:6])

            mat(compile_circuit(circ; inv = false)) == mat(chain(6, control(3:6, 1:2 => custom_block))) &&
                 mat(compile_circuit(circ; inv = true)) == mat(chain(6, control(3:6, 1:2 => custom_block_inv)))
        end
        
    end
end

@testset "Full examples" begin
    
end

@testset "Helper functions" begin
    @testset "_diffusion_gate_for_zero_function" begin
        grover_circ = empty_circuit(1, 1)
        @test GroverCircuitBuilder._diffusion_gate_for_zero_function(2, 2, true) == chain(2, put(2 => Z))
        @test GroverCircuitBuilder._diffusion_gate_for_zero_function(2, 2, false) == chain(2, put(2 => X), put(2 => Z), put(2 => X))
    end

    @testset "_convert_to_bools" begin
        @test GroverCircuitBuilder._convert_to_bools(10, [1, 2, 3, 4]) == [true, false, false, true]
    end
end