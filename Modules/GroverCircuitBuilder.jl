module GroverCircuitBuilder

    using Yao
    using ..GroverML

    export GroverBlock;
    export GroverCircuit;
    export RotationBlock;
    export HadamardBlock;
    export compile_block;
    export compile_circuit;
    export empty_circuit;
    export circuit_size;
    export model_lanes;
    export param_lanes;
    export prepare;
    export unprepare;
    export create_checkpoint;
    export manipulate_lanes;
    export insert_model_lane;
    export insert_param_lane;
    export hadamard;
    export rotation;
    export learned_rotation;
    export not;
    export yao_block;
    export build_grover_iteration;
    export auto_compute;
    export trace_inversion_problem;
    
    abstract type GroverBlock end

    mutable struct BlockMeta
        insertion_checkpoint::Int
        data::Dict
        manipulator::Union{Function, Nothing}
    end
    
    function BlockMeta(insertion_checkpoint::Int)::BlockMeta
        return BlockMeta(insertion_checkpoint, Dict(), nothing)
    end

    function BlockMeta(insertion_checkpoint::Int, data::Dict)::BlockMeta
        return BlockMeta(insertion_checkpoint, data, nothing)
    end

    mutable struct GroverCircuit
        model_lanes::Vector{Int}
        param_lanes::Vector{Int}
        circuit::Vector{<:GroverBlock}
        circuit_meta::Vector{BlockMeta}
        preparation_state::Bool
        current_checkpoint::Int
        lane_manipulators::Vector{Function}
    end

    mutable struct RotationBlock <: GroverBlock
        target_lanes::Vector{Int}
        control_lanes::Union{Vector{Vector{Int}}, Nothing}
        rotations::Vector{Number}
    end

    mutable struct HadamardBlock <: GroverBlock
        target_lanes::Vector{Int}
        control_lanes::Union{Vector{Vector{Int}}, Nothing}
    end

    mutable struct NotBlock <: GroverBlock 
        target_lanes::Vector{Int}
        control_lanes::Union{Vector{Vector{Int}}, Nothing}
    end

    mutable struct YaoBlock <: GroverBlock
        block::Yao.YaoAPI.AbstractBlock
        inv_block::Yao.YaoAPI.AbstractBlock
        target_lanes::Vector{Vector{Int}}
        control_lanes::Union{Vector{Vector{Int}}, Nothing}
    end

    mutable struct OracleBlock <: GroverBlock
        oracle_lane::Int
        target_lanes::Vector{Int}
        target_bits::Vector{Bool}
    end

    """
    Creates an empty circuit with the specified number of `model` and `param`-lanes.
    """
    function empty_circuit(model_lanes::Int, param_lanes::Int)::GroverCircuit
        return GroverCircuit(collect(1:model_lanes), collect(model_lanes+1:model_lanes+param_lanes), Vector{GroverBlock}(), Vector{BlockMeta}(), false, 1, Vector{Function}())
    end

    function circuit_size(circuit::GroverCircuit)
        return length(circuit.model_lanes) + length(circuit.param_lanes)
    end

    """
    Returns a copy of all lane indices representing the model lanes.
    If lanes have not been rearranged, these will be elements of the range `1:len(circuit.model_lanes)`.
    """
    function model_lanes(circuit::GroverCircuit)::Vector{Int}
        return circuit.model_lanes[:]
    end

    """
    Returns a copy of all lane indices representing the parameter lanes.
    If lanes have not been rearranged, these will be elements of the range `len(circuit.model_lanes)+1:circuit_size(circuit)`.
    """
    function param_lanes(circuit::GroverCircuit)::Vector{Int}
        return circuit.param_lanes[:]
    end

    """
    Automatically computes the grover-circuit to the given circuit.
    The function returns a Tuple containing the final simulated register, 
        the main circuit design without applying grover and the grover iterations.
    
    # Arguments
    - `circuit::GroverCircuit`: The main circuit.
    - `output_bits::Union{Vector, Bool}`: The output bits. This can either be a boolean value or a vector of booleans if you have more than one model lane. If you want to do batch-training, this should be a vector of output bits.

    # Optional arguments
    - `forced_grover_iterations::Union{Int, Nothing}=nothing`: Enforces a specific number of grover iterations. Otherwise the optimal number of grover iterations will be applied if possible.
    - `ignore_errors:Bool=true`: Specifies if errors should be ignored or thrown. If errors are ignored and the optimal number of grover iterations could not be determined, the function will proceed with a single grover iteration. This is helpful if you still want to extract the grover-iteration even if no optimal number of iterations could be found.
    - `evaluate:Bool=true`: Specifies if the grover circuit should actually be simulated. This is necessary to determine the optimal number of grover iterations. However, it is only possible to simulate small circuits. If you only want to build a fixed number of grover iterations, it is not necessary to simulate the state. Thus, for larger circuits it is recommended to not simulate the circuit.

    # Returns
    A tuple containing the following elements:
    - `out::Union{Yao.ArrayReg, Nothing}`: The register holding the state after applying the grover circuit if `evaluate == true`, else `nothing`
    - `main_circuit::Yao.YaoAPI.AbstractBlock`: The main circuit that was specified using the grover circuit builder
    - `grover_circuit::Yao.YaoAPI.AbstractBlock`: The generated grover circuit with either the optimal number of iterations, the number of `forced_grover_iterations` if not `nothing`, or a single iteration if `evaluate == false`
    - `oracle_function::Function`: The function that returns `true` if and only if the corresponding state index is a target state
    """
    function auto_compute(circuit::GroverCircuit, output_bits::Union{Vector, Bool}; forced_grover_iterations::Union{Int, Nothing} = nothing, ignore_errors::Bool = true, evaluate::Bool = true)::Tuple{Union{Yao.ArrayReg, Nothing}, Yao.YaoAPI.AbstractBlock, Yao.YaoAPI.AbstractBlock, Function}
        @info "Simulating grover circuit..."

        # Map the corresponding types to Vector{Int}
        target_lanes = circuit.model_lanes
        target_bits = _resolve_output_bits(output_bits)

        if length(target_bits) == 0
            throw(DomainError(target_bits, "The number of output bits must be greater than 0"))
        end

        for out_bits in target_bits
            if length(out_bits) != length(target_lanes)
                throw(DomainError(out_bits, "The number of output bits must match the number of target lanes"))
            end
        end

        oracle_lane = 1

        # If we use an additional grover-lane, we need to 
        # prepare the circuit by adding an oracle block (see function prepare(...))
        # TODO: Fix this check
        if _use_grover_lane(target_lanes) || length(target_bits) > 1
            circuit = create_checkpoint(circuit)
            # Push the oracle block to the circuit
            oracle_lane = prepare(circuit, target_lanes, target_bits)

            target_lanes = _map_lanes(circuit, circuit.current_checkpoint-1, target_lanes)
        end

        circ_size = circuit_size(circuit)

        # Initialize a zero-state (|0^(circuit_size)>)
        register = zero_state(circ_size)

        # We first create the main circuit to determine the probability of the target state
        main_circuit = compile_circuit(circuit)
        @info "Main circuit compiled"

        # Gate the zero state through the main circuit
        out = nothing
        if evaluate
            @info "Evaluating main circuit..."
            out = register |> main_circuit
            @info "Main circuit evaluated"
        end

        # Prepare the oracle function. This function is a function that returns true if the 
        # index of the respective quantum state is the target state, else false
        oracle_function = idx -> ((2^(oracle_lane-1)) & (idx-1)) != 0
        #oracle_function = _wrap_oracle(_oracle_function([oracle_lane], [true]), [oracle_lane])

        # Compute the probability of the target state after applying the main circuit
        cumulative_pre_probability = nothing 
        if evaluate 
            cumulative_pre_probability = computeCumProb(out, oracle_function)
        else
            return nothing, main_circuit, createGroverCircuit(circ_size, isnothing(forced_grover_iterations) ? 1 : forced_grover_iterations, build_grover_iteration(circuit, oracle_lane, _use_grover_lane(target_lanes) ? true : target_bits[1][1][2])), oracle_function
        end

        @info "Cumulative Pre-Probability: $cumulative_pre_probability"
        angle_rad = computeAngle(cumulative_pre_probability)
        @info "Angle towards orthogonal state: $angle_rad"
        @info "Angle towards orthogonal state (deg): $(angle_rad / pi * 180)"

        num_grover_iterations = nothing
        if ignore_errors || !isnothing(forced_grover_iterations)
            try
                # Determine the optimal number of grover iterations
                num_grover_iterations = computeOptimalGroverN(cumulative_pre_probability)
            catch err
                @error "Could not determine the optimal number of grover iterations:"
                #println("ERROR: ", e)
                @error "ERROR: " exception=(err, catch_backtrace())
            end
        else 
            # Determine the optimal number of grover iterations
            num_grover_iterations = computeOptimalGroverN(cumulative_pre_probability)
        end

        # If the we force the number of grover iterations, we need to overwrite the optimal number of grover iterations
        actual_grover_iterations = isnothing(forced_grover_iterations) ? num_grover_iterations : forced_grover_iterations

        @info "Optimal number of Grover iterations: $(isnothing(num_grover_iterations) ? "?" : num_grover_iterations)"
        @info "Actual optimum from formula: $(isnothing(num_grover_iterations) ? "?" : 1/2 * (pi/(2 * computeAngle(cumulative_pre_probability)) - 1))"

        if isnothing(actual_grover_iterations)
            @warn ""
            @warn "======== WARNING ========"
            @warn "========================="
            @warn ""
            @warn "The actual number of grover iterations could not be determined"
            @warn "We will continue using 1 grover-iterations such that the circuit can still be extracted :)"
            actual_grover_iterations = 1
        end

        @info "Compiling grover circuit..."
        # Create the grover circuit to amplify the amplitude
        grover_circuit = createGroverCircuit(circ_size, actual_grover_iterations, build_grover_iteration(circuit, oracle_lane, _use_grover_lane(target_lanes) ? true : target_bits[1][1][2]))
        @info "Grover circuit compiled"

        @info "Evaluating grover circuit..."
        # Gate the current quantum state through the grover circuit
        out = out |> grover_circuit
        @info "Grover circuit evaluated"

        cum_prob = computeCumProb(out, oracle_function)
        pred_cum_prob = computePostGroverLikelihood(computeAngle(cumulative_pre_probability), actual_grover_iterations)

        # Print the results
        @info ""
        @info "======== RESULTS ========"
        @info "========================="
        @info ""
        @info "Cumulative Probability (after $(actual_grover_iterations)x Grover): $(cum_prob)"
        @info "Predicted likelihood after $(actual_grover_iterations)x Grover: $pred_cum_prob"

        if abs(cum_prob - pred_cum_prob) > 0.01
            @warn ""
            @warn "======== WARNING ========"
            @warn "========================="
            @warn ""
            @warn "The cumulative probability does not seem to match the predicted probability!"
            @warn "Probably a wrong inverse was provided."
            trace_inversion_problem(circuit)
        end

        return out, main_circuit, grover_circuit, oracle_function
    end

    """
    This function tries to identify and isolate errors in a grover circuit. It prints out the first GroverBlock where the composition of the block and its inverse is not the identity matrix.
    If a circuit contains errors, the expected likelihood of the target state after a number of grover iterations is usually not equal to the simulated probability.
    Such errors usually appear on custom yao blocks, if the inverse block has not been specified correctly.
    """
    function trace_inversion_problem(circuit::GroverCircuit)
        @info ""
        @info "Trying to isolate the inversion problem..."

        circ_size = circuit_size(circuit)

        for state in 0:(2^circ_size - 1)
            for (idx, block) in enumerate(circuit.circuit)
                for complex in [false, true]
                    m_state = _create_array_register_from_integer(state, circ_size)

                    if complex
                        m_state = ArrayReg(im .* m_state.state)
                    end

                    m_st = copy(m_state.state)
    
                    gate = compile_block(circuit, block, circuit.circuit_meta[idx], inv = false)
                    gate_inv = compile_block(circuit, block, circuit.circuit_meta[idx], inv = true)
                    out = m_state |> gate |> gate_inv
        
                    st = out.state
    
                    err = 0
    
                    for i in 1:(state+1)
                        err += abs(st[i] - m_st[i])
                    end
                    
                    if err > 0.01
                        @warn "An error occurred at block $idx ($(typeof(block)))"
                        @warn "The inverse of the block specified does not behave as expected"
                        @warn "Cumulative error: $err"
                        @warn "State: $state"
                        return
                    end
                end
            end
        end
    end

    """
    Prepares a cricuit and inserts a target lane which is used as the oracle function.
    """
    function prepare(circuit::GroverCircuit, model_lanes::Union{AbstractRange, Vector{Int}, Int}, output_bits::Union{Vector, Bool}; insert_lane_at::Union{Int, Nothing} = nothing)::Int
        target_lanes = _resolve_lanes(model_lanes)
        target_bits = _resolve_output_bits(output_bits)

        if circuit.preparation_state
            unprepare(circuit)
        end

        flattened_target_bits = _flatten_output_bits(target_bits)

        # Extend target lanes to match batch-size
        inserted_batch_lanes = Vector{Int}()
        for idx in 2:length(target_bits)
            for i in 1:length(target_lanes)
                @info "Inserting batch-lane after lane number $((idx-1) * length(target_lanes) + i - 1)"
                insert_model_lane(circuit, (idx-1) * length(target_lanes) + i)
                push!(inserted_batch_lanes, (idx-1) * length(target_lanes) + i)
            end
        end

        # Clone corresponding model blocks
        for batch in 2:length(target_bits)
            len = length(circuit.circuit)
            idx = 1
            while idx <= len
                block = circuit.circuit[idx]
                meta = circuit.circuit_meta[idx]

                is_model_function, m_target_lanes, m_control_lanes = _check_for_batch_lanes(circuit, block, meta, target_lanes, batch, inserted_batch_lanes)

                if is_model_function
                    new_block = deepcopy(block)
                
                    new_block.target_lanes = m_target_lanes
                    new_block.control_lanes = m_control_lanes

                    insert!(circuit.circuit, idx+1, new_block)
                    new_meta = BlockMeta(circuit.current_checkpoint, copy(meta.data), meta.manipulator)
                    new_meta.data["batch"] = batch
                    insert!(circuit.circuit_meta, idx+1, new_meta)
                    idx += 1
                    len += 1
                end

                idx += 1
            end
        end

        # In order to do batch training, we need to negate each lane where the input starts with a |1>
        negate_output_lanes = Vector{Int}()
        for (b_idx, bits) in enumerate(target_bits)
            for (idx, lane) in enumerate(target_lanes)
                if bits[idx][1]
                    push!(negate_output_lanes, (b_idx-1) * length(target_lanes) + lane)
                end
            end
        end

        if length(negate_output_lanes) > 0
            pushfirst!(circuit.circuit, NotBlock(negate_output_lanes, nothing))
            pushfirst!(circuit.circuit_meta, BlockMeta(circuit.current_checkpoint))
        end

        target_lanes = vcat(target_lanes, inserted_batch_lanes)

        if isnothing(insert_lane_at)
            insert_lane_at = maximum(target_lanes) + 1
        end

        @info "Inserting grover-lane after lane number $(insert_lane_at-1)"
        insert_model_lane(circuit, insert_lane_at)
        target_lanes = _map_lanes(circuit, circuit.current_checkpoint-1, target_lanes)

        # Filter out nothing
        filtered_lanes = [target_lanes[i] for (i, val) in enumerate(flattened_target_bits) if !isnothing(val)]
        filtered_target_bits = filter(x -> !isnothing(x), flattened_target_bits)
        
        push!(circuit.circuit, OracleBlock(insert_lane_at, filtered_lanes, filtered_target_bits))
        push!(circuit.circuit_meta, BlockMeta(circuit.current_checkpoint))

        return insert_lane_at
    end

    """
    Creates a checkpoint for the circuit by deepcopying it.
    """
    function create_checkpoint(circuit::GroverCircuit)::GroverCircuit
        return deepcopy(circuit)
    end

    """
    Manipulates the order of lanes by applying a permutation (mapping function).
    This is useful if you want to rearrange lanes. 
    Note that manipulations are lazy operations which might cause confusing errors if you pass invalid mappings.
    If you insert blocks after lane manipulation you will have to use the new order.
    """
    function manipulate_lanes(circuit::GroverCircuit, mapping::Function)
        map!(mapping, circuit.model_lanes, circuit.model_lanes)
        map!(mapping, circuit.param_lanes, circuit.param_lanes)

        push!(circuit.lane_manipulators, mapping)
        circuit.current_checkpoint += 1
    end

    """
    Inserts a model-lane at a specific location. All lanes after this location will be shifted below this lane.
    """
    function insert_model_lane(circuit::GroverCircuit, location::Int)
        if location < 1 || location > circuit_size(circuit)
            throw(DomainError(location, "Target location out of bounds (must be within lanes 1:" * string(circuit_size(circuit)) * ")"))
        end
        
        manipulate_lanes(circuit, x -> x >= location ? x + 1 : x)
        push!(circuit.model_lanes, location)
    end

    """
    Inserts a param-lane at a specific location. All lanes after this location will be shifted below this lane.
    """
    function insert_param_lane(circuit::GroverCircuit, location::Int)
        if location < 1 || location > circuit_size(circuit)
            throw(DomainError(location, "Target location out of bounds (must be within lanes 1:" * string(circuit_size(circuit)) * ")"))
        end
        
        manipulate_lanes(circuit, x -> x >= location ? x + 1 : x)
        push!(circuit.param_lanes, location)
    end

    """
    Builds a single grover-iteration as a circuit.
    """
    function build_grover_iteration(circuit::GroverCircuit, target_lane::Int, target_bit::Bool)::Yao.YaoAPI.AbstractBlock
        circ_size = circuit_size(circuit)
        diff_gate = _diffusion_gate_for_zero_function(circ_size, target_lane, target_bit)
        oracle = prepareOracleGate(circ_size, compile_circuit(circuit, inv=false), compile_circuit(circuit, inv=true))
        return createGroverIteration(circ_size, diff_gate, oracle)
    end

    """
    Compiles the grover circuit as a Yao-Circuit. You can also compile the inverse circuit.
    """
    function compile_circuit(circuit::GroverCircuit; inv::Bool = false)::Yao.YaoAPI.AbstractBlock
        circ_size = circuit_size(circuit)
        m_circuit = inv ? reverse(circuit.circuit) : circuit.circuit
        m_meta = inv ? reverse(circuit.circuit_meta) : circuit.circuit_meta
        compiled_circuit = chain(circ_size)

        for (idx, block) in enumerate(m_circuit)
            compiled_block = compile_block(circuit, block, m_meta[idx], inv=inv)
            compiled_circuit = chain(circ_size, put(1:circ_size => compiled_circuit), put(1:circ_size => compiled_block))
        end

        return compiled_circuit
    end

    """
    Compiles a single Grover-Block within a circuit. You can also compile the inverse of that block.
    """
    function compile_block(circuit::GroverCircuit, block::GroverBlock, meta::BlockMeta; inv::Bool = false)::Yao.YaoAPI.AbstractBlock
        if !isnothing(meta.manipulator)
            @debug "Manipulating block..."
            if !meta.manipulator(block, meta, inv)
                @debug "Skipping block..."
                return chain(circuit_size(circuit))
            end
        end
        
        if isa(block, RotationBlock)
            return _compile_rotation_block(circuit, block::RotationBlock, meta, inv=inv)
        elseif isa(block, HadamardBlock)
            return _compile_hadamard_block(circuit, block::HadamardBlock, meta, inv=inv)
        elseif isa(block, NotBlock)
            return _compile_not_block(circuit, block::NotBlock, meta, inv=inv)
        elseif isa(block, OracleBlock)
            return _compile_oracle_block(circuit, block::OracleBlock, meta, inv=inv)
        elseif isa(block, YaoBlock)
            return _compile_yao_block(circuit, block::YaoBlock, meta, inv=inv)
        end

        throw(DomainError(block, "Unknown Grover-Block"))
    end

    """
    Creates one or multiple custom Yao-Block.

    # Examples
    ```julia
    custom_block = chain(2, put(1 => Rz(pi)), put(2 => Rz(pi)))
    custom_block_inv = chain(2, put(2 => Rz(-pi)), put(1 => Rz(-pi)))
    yao_block(grover_circ, [1:2, 1:2], custom_block, custom_block_inv, control_lanes=[3:4, 5:6])
    ```

    For more information see the [documentation](https://github.com/Jaybit0/QCProjectCode/blob/main/readme.md#custom-yao-gates).
    """
    function yao_block(circuit::GroverCircuit, target_lanes::Union{Vector, AbstractRange, Int}, block::Yao.YaoAPI.AbstractBlock, inv_block::Yao.YaoAPI.AbstractBlock; control_lanes::Union{Vector, AbstractRange, Int, Nothing} = nothing, push_to_circuit::Bool = true)::Tuple{YaoBlock, BlockMeta}
        target_lanes = _resolve_stacked_control_lanes(target_lanes)
        control_lanes = _resolve_stacked_control_lanes(control_lanes)

        block = YaoBlock(block, inv_block, target_lanes, control_lanes)
        meta = BlockMeta(circuit.current_checkpoint)

        if push_to_circuit
            push!(circuit.circuit, block)
            push!(circuit.circuit_meta, meta)
        end

        return block, meta
    end

    """
    Creates one or multiple hadamard blocks.

    # Examples 
    ```julia
    hadamard(grover_circ, 1:2, control_lanes = [3, 4:6])
    ```

    For more information see the [documentation](https://github.com/Jaybit0/QCProjectCode/blob/main/readme.md#hadamard-gates).
    """
    function hadamard(circuit::GroverCircuit, target_lanes::Union{Vector, AbstractRange, Int}; control_lanes::Union{Vector, AbstractRange, Int, Nothing} = nothing, push_to_circuit::Bool = true)::Tuple{HadamardBlock, BlockMeta}
        target_lanes = _resolve_lanes(target_lanes)
        control_lanes = _resolve_stacked_control_lanes(control_lanes)
        
        block = HadamardBlock(target_lanes, control_lanes)
        meta = BlockMeta(circuit.current_checkpoint)

        if push_to_circuit
            push!(circuit.circuit, block)
            push!(circuit.circuit_meta, meta)
        end

        return block, meta
    end

    """
    Creates a learned rotation.

    # Examples

    ```julia
    learned_rotation(grover_circ, 1, 3:6)
    ```

    For more information see the [documentation](https://github.com/Jaybit0/QCProjectCode/blob/main/readme.md#learned-rotation-gates).
    """
    function learned_rotation(circuit::GroverCircuit, target_lane::Int, control_lanes::Union{Vector, AbstractRange, Int}; max_rotation_rad::Number = 2*pi, push_to_circuit::Bool = true)::Tuple{RotationBlock, BlockMeta}
        control_lanes = _resolve_stacked_control_lanes(control_lanes)

        current_rotation = max_rotation_rad
        rotations = Vector{Number}()

        for _ in control_lanes
            current_rotation /= 2
            push!(rotations, current_rotation)
        end

        block = RotationBlock([target_lane], control_lanes, rotations)
        meta = BlockMeta(circuit.current_checkpoint)

        if push_to_circuit
            push!(circuit.circuit, block)
            push!(circuit.circuit_meta, meta)
        end

        return block, meta
    end

    """
    A legacy rotation.
    """
    function rotation(circuit::GroverCircuit, target_lane::Int; control_lanes::Union{Vector, AbstractRange, Int, Nothing} = nothing, max_rotation_rad::Number = 2*pi, push_to_circuit::Bool = true)::Tuple{RotationBlock, BlockMeta}
        control_lanes = _resolve_stacked_control_lanes(control_lanes)
        
        granularity = isnothing(control_lanes) ? 1 : length(control_lanes)
        angle = max_rotation_rad / granularity

        rotations = Vector{Number}()

        if typeof(control_lanes) <: AbstractRange
            m_control_lanes = Vector{AbstractRange}()

            for i in control_lanes
                push!(m_control_lanes, i:i)
            end

            control_lanes = m_control_lanes
        end

        if isnothing(control_lanes)
            push!(rotations, angle)
        else
            for i in control_lanes
                push!(rotations, angle)
            end
        end

        block = RotationBlock([target_lane], control_lanes, rotations)
        meta = BlockMeta(circuit.current_checkpoint)

        if push_to_circuit
            push!(circuit.circuit, block)
            push!(circuit.circuit_meta, meta)
        end

        return block, meta
    end

    """
    Creates one or multiple not-gates.

    # Examples
    ```julia
    not(grover_circ, 1:2, control_lanes = [3, 4:6])
    ```

    For more information see the [documentation](https://github.com/Jaybit0/QCProjectCode/blob/main/readme.md#not-gates).
    """
    function not(circuit::GroverCircuit, target_lanes::Union{Vector, AbstractRange, Int}; control_lanes::Union{Int, AbstractRange, Vector, Nothing} = nothing, push_to_circuit::Bool = true)::Tuple{NotBlock, BlockMeta}
        target_lanes = _resolve_lanes(target_lanes)
        control_lanes = _resolve_stacked_control_lanes(control_lanes)

        block = NotBlock(target_lanes, control_lanes)
        meta = BlockMeta(circuit.current_checkpoint)

        if push_to_circuit
            push!(circuit.circuit, block)
            push!(circuit.circuit_meta, meta)
        end

        return block, meta
    end




    # ===== Compilation functions =====
    # =================================

    function _compile_rotation_block(circuit::GroverCircuit, block::RotationBlock, meta::BlockMeta; inv::Bool = false)::Yao.YaoAPI.AbstractBlock
        circ_size = circuit_size(circuit)
        
        if isnothing(block.control_lanes)
            #TODO: Is mapping correct?
            return chain(circ_size, put(_map_lanes(circuit, meta.insertion_checkpoint, block.target_lanes) => Ry(inv ? -block.rotations[1] : block.rotations[1])))
        end

        m_circuit = chain(circ_size)
        m_rotations = inv ? reverse(block.rotations) : block.rotations

        m_control_gates = inv ? reverse(block.control_lanes) : block.control_lanes

        for i in eachindex(m_control_gates)
            angle = m_rotations[i]
            gates = m_control_gates[i]
            m_circuit = chain(circ_size, put(1:circ_size => m_circuit), control(_map_lanes(circuit, meta.insertion_checkpoint, gates), _map_lanes(circuit, meta.insertion_checkpoint, block.target_lanes) => Ry(inv ? -angle : angle)))
        end

        return m_circuit
    end

    function _compile_hadamard_block(circuit::GroverCircuit, block::HadamardBlock, meta::BlockMeta; inv::Bool = false)::Yao.YaoAPI.AbstractBlock
        circ_size = circuit_size(circuit)
        
        if isnothing(block.control_lanes)
            return chain(circ_size, repeat(H, _map_lanes(circuit, meta.insertion_checkpoint, block.target_lanes)))
        end

        if length(block.control_lanes) != length(block.target_lanes)
            throw(DomainError(block.control_lanes, "Number of target- and control-lanes do not match!"))
        end

        m_target_lanes = _map_lanes(circuit, meta.insertion_checkpoint, block.target_lanes)
        m_control_lanes = block.control_lanes

        if inv
            m_target_lanes = reverse(m_target_lanes)
            m_control_lanes = reverse(m_control_lanes)            
        end

        m_circuit = chain(circ_size)
        
        i = 1
        for target in m_target_lanes
            ctrl = _map_lanes(circuit, meta.insertion_checkpoint, m_control_lanes[i])
            m_circuit = chain(circ_size, put(1:circ_size => m_circuit), control(ctrl, target => H))
            i += 1
        end

        return m_circuit
    end

    function _compile_not_block(circuit::GroverCircuit, block::NotBlock, meta::BlockMeta; inv::Bool = false)::Yao.YaoAPI.AbstractBlock
        circ_size = circuit_size(circuit)

        if length(block.target_lanes) == 0
            return chain(circ_size)
        end

        if isnothing(block.control_lanes)
            # TODO: Handle inv
            return chain(circ_size, repeat(X, _map_lanes(circuit, meta.insertion_checkpoint, block.target_lanes)))
        end

        if length(block.control_lanes) != length(block.target_lanes)
            throw(DomainError(block.control_lanes, "Number of target- and control-lanes do not match!"))
        end

        m_target_lanes = _map_lanes(circuit, meta.insertion_checkpoint, block.target_lanes)
        m_control_lanes = block.control_lanes

        if inv
            m_target_lanes = reverse(m_target_lanes)
            m_control_lanes = reverse(m_control_lanes)            
        end

        m_circuit = chain(circ_size)

        i = 1
        for target in m_target_lanes
            ctrl = _map_lanes(circuit, meta.insertion_checkpoint, m_control_lanes[i])
            m_circuit = chain(circ_size, put(1:circ_size => m_circuit), control(ctrl, target => X))
            i += 1
        end

        return m_circuit
    end

    function _compile_yao_block(circuit::GroverCircuit, block::YaoBlock, meta::BlockMeta; inv::Bool = false)::Yao.YaoAPI.AbstractBlock
        circ_size = circuit_size(circuit)

        if isnothing(block.control_lanes)
            lanes = _map_lanes(circuit, meta.insertion_checkpoint, block.target_lanes)
            if inv
                lanes = reverse(lanes)
            end

            m_circ = chain(circ_size)

            for l in lanes
                m_circ = chain(circ_size, put(1:circ_size => m_circ), put(l => inv ? block.inv_block : block.block))
            end
            return m_circ
        end

        m_target_lanes = _map_lanes(circuit, meta.insertion_checkpoint, block.target_lanes)
        m_control_lanes = _map_lanes(circuit, meta.insertion_checkpoint, block.control_lanes)

        if inv 
            m_target_lanes = reverse(m_target_lanes)
            m_control_lanes = reverse(m_control_lanes)
        end

        m_circuit = chain(circ_size)

        for (idx, target) in enumerate(m_target_lanes)
            ctrl = m_control_lanes[idx]
            m_circuit = chain(circ_size, put(1:circ_size => m_circuit), control(ctrl, target => (inv ? block.inv_block : block.block)))
        end

        return m_circuit
    end

    function _compile_oracle_block(circuit::GroverCircuit, block::OracleBlock, meta::BlockMeta; inv::Bool = false)::Yao.YaoAPI.AbstractBlock
        circ_size = circuit_size(circuit)

        m_target_lanes = _map_lanes(circuit, meta.insertion_checkpoint, block.target_lanes)

        enum = inv ? reverse(collect(enumerate(m_target_lanes))) : enumerate(m_target_lanes)
        x_gates = chain(circ_size)

        for (idx, lane) in enum
            if !block.target_bits[idx]
                x_gates = chain(circ_size, put(1:circ_size => x_gates), put(lane => X))
            end
        end

        return chain(circ_size, put(1:circ_size => x_gates), control(m_target_lanes, _map_lanes(circuit, meta.insertion_checkpoint, block.oracle_lane) => X), put(1:circ_size => x_gates))
    end


    # ===== HELPER FUNCTIONS =====
    # ============================

    """
        _diffusion_gate_for_zero_function(circuitSize::Int, target_lane::Int, target_bit::Bool)::Yao.YaoAPI.AbstractBlock)

    Creates a diffusion gate for the given target lane and target bit. The diffusion gate is a gate that inverts the amplitude of the target state.
    We assume that the input state is |0>.
        
    """
    function _diffusion_gate_for_zero_function(circuitSize::Int, target_lane::Int, target_bit::Bool)::Yao.YaoAPI.AbstractBlock
        if target_bit
            return chain(circuitSize, put(target_lane => Z))
        else
            return chain(circuitSize, put(target_lane => X), put(target_lane => Z), put(target_lane => X))
        end
    end

    function _wrap_oracle(oracle::Function, outRange::Vector{Int})
        return idx -> oracle(_convert_to_bools(idx, outRange))
    end

    """
        _convert_to_bools(index::Int, outRange::Vector{Int})::Vector{Bool}

    Converts the given index substracted by 1 to a boolean array.
    The index is assumed to be in the range of 0 to 2^length(outRange)-1.

    """
    function _convert_to_bools(index::Int, outRange::Vector{Int})::Vector{Bool}
        out = falses(length(outRange))
        index -= 1 # To adjust the offset in the quantum register (state |0> = 1)
    
        for (c, i) in enumerate(outRange)
            out[c] = ((index >> (i - 1)) & 1) == 1
        end
    
        return out
    end

    function _oracle_function(target_lanes::Vector{Int}, target_bits::Vector{Bool})::Function
        f =  function(data)
            for (idx, _) in enumerate(target_lanes)
                if target_bits[idx] != data[idx]
                    return false
                end
            end

            return true
        end
    end

    function _map_lanes(circuit::GroverCircuit, checkpoint::Int, lanes::Vector{Int})::Vector{Int}
        for i in checkpoint:length(circuit.lane_manipulators)
            lanes = map(circuit.lane_manipulators[i], lanes)
        end

        return lanes
    end

    function _map_lanes(circuit::GroverCircuit, checkpoint::Int, lanes::Vector{Vector{Int}})::Vector{Vector{Int}}
        out = Vector{Vector{Int}}()

        for l in lanes
            push!(out, _map_lanes(circuit, checkpoint, l))
        end

        return out
    end

    function _map_lanes(circuit::GroverCircuit, checkpoint::Int, lane::Int)::Int
        return _map_lanes(circuit, checkpoint, [lane])[1]
    end

    function _use_grover_lane(target_lanes::Vector{Int})
        return length(target_lanes) > 1
    end

    function _resolve_lanes(data::Union{AbstractRange, Vector, Int})::Vector{Int}
        if isa(data, Int)
            return [data]::Vector{Int}
        end

        if typeof(data) <: AbstractRange
            return collect(Int, data)
        end

        if typeof(data) <: Vector{Int}
            return data
        end

        return Int.(data)
    end

    function _resolve_stacked_control_lanes(data::Union{Vector{Vector{Int}}, Vector{Int}, Vector, AbstractRange, Int, Nothing})::Union{Vector{Vector{Int}}, Nothing}
        if isa(data, Vector{Vector{Int}})
            return data
        end

        if isa(data, Vector) || typeof(data) <: AbstractRange
            return map(_resolve_lanes, data)
        end

        if isa(data, Int)
            return [[data]]
        end

        if isa(data, Nothing)
            return nothing
        end

        throw(DomainError(data, "Given data type is not allowed!"))
    end

    using Yao

    function _create_array_register_from_integer(value::Integer, num_qubits::Int)
        # Ensure the value fits within the specified number of qubits
        if value >= 2^num_qubits
            error("The integer is too large to fit in the specified number of qubits")
        end
    
        # Create the ArrayRegister with the correct bit configuration
        return product_state(num_qubits, value)
    end

    function _resolve_output_bits(output_bits)::Vector{Vector{Tuple{Bool, Union{Bool, Nothing}}}}
        if isa(output_bits, Vector{Vector{Tuple{Bool, Union{Bool, Nothing}}}})
            return output_bits
        end
        

        if isa(output_bits, Vector)
            if isa(output_bits[1], Vector)
                out = Vector{Vector{Tuple{Bool, Union{Bool, Nothing}}}}()
                for output in output_bits
                    push!(out, map(_try_map_output, output))
                end
                return out
            end
            out = Vector{Vector{Tuple{Bool, Union{Bool, Nothing}}}}()
            push!(out, map(_try_map_output, output_bits))
            return out
        end

        if isa(output_bits, Tuple{Bool, Union{Bool, Nothing}})
            return [[output_bits]]
        end

        if isa(output_bits, Bool)
            return [[(false, output_bits)]]
        end

        throw(DomainError(output_bits, "The given output bits are not allowed!"))
    end

    function _try_map_output(output_bit)::Tuple{Bool, Union{Bool, Nothing}}
        if isa(output_bit, Tuple{Bool, Union{Bool, Nothing}})
            return output_bit
        end

        if isa(output_bit, Tuple{Bool, Bool})
            return (output_bit[1], output_bit[2])
        end

        if isa(output_bit, Union{Bool, Nothing})
            return (false, output_bit)
        end

        if isa(output_bit, Bool)
            return (false, output_bit)
        end

        if isnothing(output_bit)
            return (false, nothing)
        end

        throw(DomainError(output_bit, "The given output bit is not allowed!"))
    end

    function _flatten_output_bits(output_bits::Vector{Vector{Tuple{Bool, Union{Bool, Nothing}}}})::Vector{Union{Bool, Nothing}}
        return reduce(vcat, map(inner_vec -> map(x -> x[2], inner_vec), output_bits))
    end

    function _check_for_batch_lanes(circuit::GroverCircuit, block::GroverBlock, meta::BlockMeta, model_lanes::Vector{Int}, batch::Int, inserted_batch_lanes::Vector{Int})::Tuple{Bool, Vector{Int}, Union{Vector{Vector{Int}}, Nothing}}
        m_batch_lane = false

        m_target_lanes = _map_lanes(circuit, meta.insertion_checkpoint, block.target_lanes)
        m_control_lanes = isnothing(block.control_lanes) ? nothing : _map_lanes(circuit, meta.insertion_checkpoint, block.control_lanes)

        # Walk through individual target lanes and offset them if they are in a batch lane
        for (i, lane) in enumerate(m_target_lanes)
            for (j, batch_lane) in enumerate(model_lanes)
                if lane == batch_lane
                    m_batch_lane = true
                    m_target_lanes[i] = inserted_batch_lanes[(batch-2) * length(model_lanes) + j]
                end
            end
        end

        # Walk through individual control lanes and offset them if they are in a batch lane
        if !isnothing(block.control_lanes)
            for (i, ctrl) in enumerate(block.control_lanes)
                for (i2, lane) in enumerate(ctrl)
                    for (j, batch_lane) in enumerate(model_lanes)
                        if lane == batch_lane
                            m_batch_lane = true
                            m_control_lanes[i][i2] = inserted_batch_lanes[(batch-2) * length(model_lanes) + j]
                        end
                    end
                end
            end
        end

        return m_batch_lane, m_target_lanes, m_control_lanes
    end
end