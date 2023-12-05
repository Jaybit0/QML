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
    export target_lanes;
    export model_lanes;
    export prepare;
    export unprepare;
    export create_checkpoint;
    export manipulate_lanes;
    export insert_target_lane;
    export hadamard;
    export rotation;
    export learned_rotation;
    export not;
    export yao_block;
    export build_grover_iteration;
    export auto_compute;
    export trace_inversion_problem;
    export _resolve_output_bits;
    
    abstract type GroverBlock end

    mutable struct BlockMeta
        insertion_checkpoint::Int
    end

    mutable struct GroverCircuit
        target_lanes::Vector{Int}
        model_lanes::Vector{Int}
        circuit::Vector{<:GroverBlock}
        circuit_meta::Vector{BlockMeta}
        preparation_state::Bool
        current_checkpoint::Int
        lane_manipulators::Vector{Function}
    end

    mutable struct RotationBlock <: GroverBlock
        target_lanes::Int
        control_lanes::Union{Vector{Vector{Int}}, Nothing}
        rotations::Vector{Number}
        is_model_function::Bool
    end

    mutable struct HadamardBlock <: GroverBlock
        target_lanes::Vector{Int}
        control_lanes::Union{Vector{Vector{Int}}, Nothing}
        is_model_function::Bool
    end

    mutable struct NotBlock <: GroverBlock 
        target_lanes::Vector{Int}
        control_lanes::Union{Vector{Vector{Int}}, Nothing}
        is_model_function::Bool
    end

    mutable struct YaoBlock <: GroverBlock
        block::Yao.YaoAPI.AbstractBlock
        inv_block::Yao.YaoAPI.AbstractBlock
        target_lanes::Vector{Vector{Int}}
        control_lanes::Union{Vector{Vector{Int}}, Nothing}
        is_model_function::Bool
    end

    mutable struct OracleBlock <: GroverBlock
        oracle_lane::Int
        target_lanes::Vector{Int}
        target_bits::Vector{Bool}
        is_model_function::Bool
    end

    function empty_circuit(target_lanes::Int, model_lanes::Int)::GroverCircuit
        return GroverCircuit(collect(1:target_lanes), collect(target_lanes+1:target_lanes+model_lanes), Vector{GroverBlock}(), Vector{BlockMeta}(), false, 1, Vector{Function}())
    end

    function circuit_size(circuit::GroverCircuit)
        return length(circuit.target_lanes) + length(circuit.model_lanes)
    end

    function target_lanes(circuit::GroverCircuit)::Vector{Int}
        return circuit.target_lanes[:]
    end

    function model_lanes(circuit::GroverCircuit)::Vector{Int}
        return circuit.model_lanes[:]
    end

    function auto_compute(circuit::GroverCircuit, output_lanes::Union{AbstractRange, Vector{Int}, Int}, output_bits::Union{Vector, Bool}; forced_grover_iterations::Union{Int, Nothing} = nothing, ignore_errors::Bool = true, evaluate::Bool = true)::Tuple{Union{Yao.ArrayReg, Nothing}, Yao.YaoAPI.AbstractBlock, Yao.YaoAPI.AbstractBlock}
        @info "Simulating grover circuit..."

        # Map the corresponding types to Vector{Int}
        target_lanes = _resolve_lanes(output_lanes)
        target_bits = _resolve_output_bits(output_bits)

        for out_bits in target_bits
            if length(out_bits) != length(target_lanes)
                throw(DomainError(out_bits, "The number of output bits must match the number of target lanes"))
            end
        end

        oracle_lane = 1

        # If we use an additional grover-lane, we need to 
        # prepare the circuit by adding an oracle block (see function prepare(...))
        if _use_grover_lane(target_lanes)
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

        # Gate the zero state through the main circuit
        out = nothing
        if evaluate
            out = register |> main_circuit
        end

        # Prepare the oracle function. This function is a function that returns true if the 
        # index of the respective quantum state is the target state, else 0
        # TODO: Simplify this by utilizing the grover lane
        oracle_function = _wrap_oracle(_oracle_function([oracle_lane], [true]), [oracle_lane])
        # Compute the probability of the target state after applying the main circuit
        cumulative_pre_probability = nothing 
        if evaluate 
            cumulative_pre_probability = computeCumProb(out, oracle_function)
        else

            return nothing, main_circuit, createGroverCircuit(circ_size, isnothing(forced_grover_iterations) ? 1 : forced_grover_iterations, build_grover_iteration(circuit, oracle_lane, _use_grover_lane(target_lanes) ? true : target_bits[1]))
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

        # Create the grover circuit to amplify the amplitude
        grover_circuit = createGroverCircuit(circ_size, actual_grover_iterations, build_grover_iteration(circuit, oracle_lane, _use_grover_lane(target_lanes) ? true : target_bits[1]))

        # Gate the current quantum state through the grover circuit
        out = out |> grover_circuit

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

        return out, main_circuit, grover_circuit
    end

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

    function prepare(circuit::GroverCircuit, output_lanes::Union{AbstractRange, Vector{Int}, Int}, output_bits::Union{Vector, Bool}; insert_lane_at::Union{Int, Nothing} = nothing)::Int
        target_lanes = _resolve_lanes(output_lanes)
        target_bits = _resolve_output_bits(output_bits)

        if isnothing(insert_lane_at)
            insert_lane_at = maximum(target_lanes) + 1
        end

        if circuit.preparation_state
            unprepare(circuit)
        end

        flattened_target_bits = _flatten_output_bits(target_bits)

        @info "Inserting grover-lane after lane number $(insert_lane_at-1)"
        insert_target_lane(circuit, insert_lane_at)
        target_lanes = _map_lanes(circuit, circuit.current_checkpoint-1, target_lanes)
        
        push!(circuit.circuit, OracleBlock(insert_lane_at, target_lanes, flattened_target_bits, true))
        push!(circuit.circuit_meta, BlockMeta(circuit.current_checkpoint))

        # Extend target lanes to match batch-size
        inserted_batch_lanes = Vector{Int}()
        for idx in 2:length(target_bits)
            for i in 1:length(target_lanes)
                @info "Inserting batch-lane after lane number $((idx-1) * length(target_lanes) + i)"
                insert_target_lane(circuit, (idx-1) * length(target_lanes) + i)
                push!(inserted_batch_lanes, (idx-1) * length(target_lanes) + i)
            end
        end

        # Clone corresponding model blocks
        for batch in 2:length(target_bits)
            len = length(circuit.circuit)
            idx = 1
            while idx <= len
                block = circuit.circuit[idx]
                @info block
                if block.is_model_function
                    # TODO: This should be reworked
                    if isa(block, OracleBlock)
                        block.target_lanes = _map_lanes(circuit, circuit.circuit_meta[idx].insertion_checkpoint, block.target_lanes)
                        block.oracle_lane = _map_lanes(circuit, circuit.circuit_meta[idx].insertion_checkpoint, block.oracle_lane)
                        circuit.circuit_meta[idx].insertion_checkpoint = circuit.current_checkpoint

                        append!(block.target_lanes, inserted_batch_lanes)
                    else
                        new_block = deepcopy(block)
                    
                        new_block.target_lanes = _map_lanes(circuit, circuit.circuit_meta[idx].insertion_checkpoint, new_block.target_lanes)
                        new_block.control_lanes = _map_lanes(circuit, circuit.circuit_meta[idx].insertion_checkpoint, new_block.control_lanes)
    
                        # Trivially offset target lanes. 
                        if isa(new_block, RotationBlock)
                            new_block.target_lanes += (batch-1) * length(target_lanes)
                        else
                            new_block.target_lanes = map(x -> x + (batch-1) * length(target_lanes), new_block.target_lanes)
                        end
    
                        insert!(circuit.circuit, idx+1, new_block)
                        insert!(circuit.circuit_meta, idx+1, BlockMeta(circuit.current_checkpoint))
                        idx += 1
                        len += 1
                    end
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

        pushfirst!(circuit.circuit, NotBlock(negate_output_lanes, nothing, false))
        pushfirst!(circuit.circuit_meta, BlockMeta(circuit.current_checkpoint))

        return insert_lane_at
    end

    function create_checkpoint(circuit::GroverCircuit)::GroverCircuit
        return deepcopy(circuit)
    end

    function manipulate_lanes(circuit::GroverCircuit, mapping::Function)
        map!(mapping, circuit.target_lanes, circuit.target_lanes)
        map!(mapping, circuit.model_lanes, circuit.target_lanes)

        push!(circuit.lane_manipulators, mapping)
        circuit.current_checkpoint += 1
    end

    function insert_target_lane(circuit::GroverCircuit, location::Int)
        if location < 1 || location > circuit_size(circuit)
            throw(DomainError(location, "Target location out of bounds (must be within lanes 1:" * string(circuit_size(circuit)) * ")"))
        end
        
        manipulate_lanes(circuit, x -> x >= location ? x + 1 : x)
        push!(circuit.target_lanes, location)
    end

    function build_grover_iteration(circuit::GroverCircuit, oracle_lane::Int, target_bit::Bool)::Yao.YaoAPI.AbstractBlock
        circ_size = circuit_size(circuit)
        diff_gate = _diffusion_gate_for_zero_function(circ_size, oracle_lane, target_bit)
        oracle = prepareOracleGate(circ_size, compile_circuit(circuit, inv=false), compile_circuit(circuit, inv=true))
        return createGroverIteration(circ_size, diff_gate, oracle)
    end

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

    function compile_block(circuit::GroverCircuit, block::GroverBlock, meta::BlockMeta; inv::Bool = false)::Yao.YaoAPI.AbstractBlock
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

    function yao_block(circuit::GroverCircuit, target_lanes::Union{Vector, AbstractRange, Int}, block::Yao.YaoAPI.AbstractBlock, inv_block::Yao.YaoAPI.AbstractBlock, is_model_function::Bool; control_lanes::Union{Vector, AbstractRange, Int, Nothing} = nothing, push_to_circuit::Bool = true)::Tuple{YaoBlock, BlockMeta}
        target_lanes = _resolve_stacked_control_lanes(target_lanes)
        control_lanes = _resolve_stacked_control_lanes(control_lanes)

        block = YaoBlock(block, inv_block, target_lanes, control_lanes, is_model_function)
        meta = BlockMeta(circuit.current_checkpoint)

        if push_to_circuit
            push!(circuit.circuit, block)
            push!(circuit.circuit_meta, meta)
        end

        return block, meta
    end

    function hadamard(circuit::GroverCircuit, target_lanes::Union{Vector, AbstractRange, Int}, is_model_function::Bool; control_lanes::Union{Vector, AbstractRange, Int, Nothing} = nothing, push_to_circuit::Bool = true)::Tuple{HadamardBlock, BlockMeta}
        target_lanes = _resolve_lanes(target_lanes)
        control_lanes = _resolve_stacked_control_lanes(control_lanes)
        
        block = HadamardBlock(target_lanes, control_lanes, is_model_function)
        meta = BlockMeta(circuit.current_checkpoint)

        if push_to_circuit
            push!(circuit.circuit, block)
            push!(circuit.circuit_meta, meta)
        end

        return block, meta
    end

    function learned_rotation(circuit::GroverCircuit, target_lane::Int, control_lanes::Union{Vector, AbstractRange, Int}, is_model_function::Bool; max_rotation_rad::Number = 2*pi, push_to_circuit::Bool = true)::Tuple{RotationBlock, BlockMeta}
        control_lanes = _resolve_stacked_control_lanes(control_lanes)

        current_rotation = max_rotation_rad
        rotations = Vector{Number}()

        for i in control_lanes
            current_rotation /= 2
            push!(rotations, current_rotation)
        end

        block = RotationBlock(target_lane, control_lanes, rotations, is_model_function)
        meta = BlockMeta(circuit.current_checkpoint)

        if push_to_circuit
            push!(circuit.circuit, block)
            push!(circuit.circuit_meta, meta)
        end

        return block, meta
    end

    function rotation(circuit::GroverCircuit, target_lane::Int, is_model_function::Bool; control_lanes::Union{Vector, AbstractRange, Int, Nothing} = nothing, max_rotation_rad::Number = 2*pi, push_to_circuit::Bool = true)::Tuple{RotationBlock, BlockMeta}
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

        block = RotationBlock(target_lane, control_lanes, rotations, is_model_function)
        meta = BlockMeta(circuit.current_checkpoint)

        if push_to_circuit
            push!(circuit.circuit, block)
            push!(circuit.circuit_meta, meta)
        end

        return block, meta
    end

    function not(circuit::GroverCircuit, target_lanes::Union{Vector, AbstractRange, Int}, is_model_function::Bool; control_lanes::Union{Int, AbstractRange, Vector, Nothing} = nothing, push_to_circuit::Bool = true)::Tuple{NotBlock, BlockMeta}
        target_lanes = _resolve_lanes(target_lanes)
        control_lanes = _resolve_stacked_control_lanes(control_lanes)

        block = NotBlock(target_lanes, control_lanes, is_model_function)
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
            return chain(circ_size, put(_resolve_grover_lane(circuit, block.target_lanes) => Ry(inv ? -block.rotations[1] : block.rotations[1])))
        end

        m_circuit = chain(circ_size)
        m_rotations = inv ? reverse(block.rotations) : block.rotations

        m_control_gates = inv ? reverse(block.control_lanes) : block.control_lanes

        for i in 1:length(m_control_gates)
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
        
        if !_use_grover_lane(block.target_lanes)
            return chain(circ_size) # return identity
        end

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
            for (idx, lane) in enumerate(target_lanes)
                if target_bits[idx] != data[idx]
                    return false
                end
            end

            return true
        end
    end

    function _map_lanes(circuit::GroverCircuit, checkpoint::Int, lanes::Vector{Int})::Vector{Int}
        for i in checkpoint:length(circuit.lane_manipulators)
            #println("Mapping: ", i)
            lanes = map(circuit.lane_manipulators[i], lanes)
            #println(lanes)
        end

        #println("Grover:")
        #lanes = _resolve_grover_lanes(circuit, lanes)
        #println(lanes)

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

    function _resolve_lanes(data::Union{AbstractRange, Vector{Int}, Int})::Vector{Int}
        if isa(data, Int)
            return [data]::Vector{Int}
        end

        if typeof(data) <: AbstractRange
            return collect(Int, data)
        end

        return data
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

    function _resolve_output_bits(output_bits)::Vector{Vector{Tuple{Bool, Bool}}}
        if isa(output_bits, Vector{Vector{Tuple{Bool, Bool}}})
            return output_bits
        end
        if isa(output_bits, Vector{Vector{Bool}})
            out = Vector{Vector{Tuple{Bool, Bool}}}()
            for output in output_bits
                push!(out, map(x -> (false, x), output))
            end
            return out
        end

        if isa(output_bits, Vector{Bool})
            out = Vector{Vector{Tuple{Bool, Bool}}}()
            push!(out, map(x -> (false, x), output_bits))
            return out
        end

        if isa(output_bits, Vector{Tuple{Bool, Bool}})
            return [output_bits]
        end

        if isa(output_bits, Tuple{Bool, Bool})
            return [[output_bits]]
        end

        if isa(output_bits, Bool)
            return [[(false, output_bits)]]
        end

        throw(DomainError(output_bits, "The given output bits are not allowed!"))
    end

    function _flatten_output_bits(output_bits::Vector{Vector{Tuple{Bool, Bool}}})::Vector{Bool}
        return reduce(vcat, map(inner_vec -> map(x -> x[2], inner_vec), output_bits))
    end
end