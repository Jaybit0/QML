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
using Logging

Base.global_logger(ConsoleLogger(stdout, Logging.Warn))

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
    @test begin
        grover_circ = empty_circuit(1, 4)

        hadamard(grover_circ, 2:5)

        rotation(grover_circ, model_lanes(grover_circ)[1], control_lanes = 2:5; max_rotation_rad = 2*pi)

        criterion = [true]
        out, main_circ, grov, oracle_function = auto_compute(grover_circ, criterion; ignore_errors = false)
        true
    end

    @test begin
        # Initialize an empty circuit with 2 target lanes and 4 model lanes
        grover_circ = empty_circuit(2, 4)

        # Apply Hadamard Gates on the lanes 3 -> 6
        hadamard(grover_circ, 3:6)

        # Apply 3 controlled rotations on the first lane with a granularity of pi/4 (max_rotation_rad / 2^length(control_lanes))
        block, meta = learned_rotation(grover_circ, 1, 3:5)
        meta.data["batch"] = 1
        meta.data["lane"] = 1
        meta.manipulator(block, meta, inv) = meta.data["batch"] == 1
        # Apply 1 controlled rotation on the second lane with a granularity of pi (max_rotation_rad / 2^length(control_lanes))
        learned_rotation(grover_circ, 2, 6)

        # Apply a controlled negation to the second lane
        not(grover_circ, 2; control_lanes = 1)

        # We expect the first lane to return true and the second lane to return false
        # As we use multiple target lanes, auto_compute automatically inserts a lane below the target lanes which encode the criterions to this lane
        # The reflection is done with respect to the inserted lane
        criterion = [true, true]
        out, main_circ, grov, oracle_function = auto_compute(grover_circ, criterion; evaluate = true, ignore_errors = false)
        true
    end
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