function remap(circ::Yao.PutBlock, mapping::Dict, expected_length::Int=Yao.nqubits(circ))::Tuple{Union{Union{Yao.AbstractBlock, Function}, Nothing}, Bool}     
    @debug ""
    @debug "===== PUT ====="
    @debug circ
    # Here we need to insert a new location to put according to the mapping
    insert_locs = []
    true_locs = []

    # Enumerate through all locations
    for (_, loc) in enumerate(circ.locs)
        
        if haskey(mapping, loc)
            # If the location is in the mapping, we need to insert a new location
            # according to the mapping
            append!(insert_locs, mapping[loc])
            append!(true_locs, loc)
        else
            append!(true_locs, loc)
        end
    end

    if length(insert_locs) == 0 
        @debug "NOTHING TO DO"
        return nothing, true
    elseif length(insert_locs) != length(circ.locs)
        @debug "INJECTION BY REPLACEMENT"
        @debug "RECURSIVE CALL REQUIRED"
        # Copy mapping and rearrange
        new_mapping = Dict()
        new_locs = []
        append!(new_locs, circ.locs...)

        inserted = false

        for (i, loc) in enumerate(circ.locs)
            if haskey(mapping, loc)
                # Find index of the location where circ.locs[j] == mapping[loc]
                # If it is not found, then we need to insert a location in circ.locs

                mapped_loc = findfirst(x -> x == mapping[loc], circ.locs)

                if isnothing(mapped_loc)
                    append!(new_locs, mapping[loc])
                    new_mapping[i] = length(new_locs)
                    inserted = true
                else
                    new_mapping[i] = mapped_loc
                end
            end
        end

        new_put = circ
        if inserted
            @debug "DIMENSION MISMATCH: RECOMPILATION REQUIRED"
            cpy = Yao.copy(Yao.subblocks(circ)[1])
            new_cpy, _ = remap(cpy, new_mapping, length(new_locs))
            new_put = Yao.put(new_locs => new_cpy)(length(new_locs))
        end
        return new_put, false
    elseif length(insert_locs) != expected_length
        @debug "DIMENSION MISMATCH: RECOMPILATION REQUIRED"
        return Yao.put(insert_locs => Yao.copy(Yao.subblocks(circ)[1]))(expected_length), true
    else
        if length(Yao.subblocks(circ)) != 1
            throw(ArgumentError("This module does not yet support multiple subblocks"))
        end

        @debug "INJECTION BY DUPLICATION"
        
        return Yao.put(insert_locs => Yao.copy(Yao.subblocks(circ)[1]))(Yao.nqubits(circ)), true
    end
end

function remap(circ::Yao.ControlBlock, mapping::Dict, expected_length::Int=Yao.nqubits(circ))::Tuple{Union{Union{Yao.AbstractBlock, Function}, Nothing}, Bool}
    @debug ""
    @debug "===== CONTROL ====="
    @debug circ

    # Here we need to insert a new location to put according to the mapping
    insert_locs = []

    # Enumerate through all locations
    for (_, loc) in enumerate(circ.locs)
        
        if haskey(mapping, loc)
            # If the location is in the mapping, we need to insert a new location
            # according to the mapping
            append!(insert_locs, mapping[loc])
        end
        
    end

    # Here we need to insert a new location to put according to the mapping
    new_ctrl_locs = []

    # Enumerate through all locations
    for (_, loc) in enumerate(circ.ctrl_locs)
        
        if haskey(mapping, loc)
            # If the location is in the mapping, we need to insert a new location
            # according to the mapping
            append!(new_ctrl_locs, mapping[loc])
        else
            append!(new_ctrl_locs, loc)
        end
        
    end

    if length(insert_locs) == 0 
        @debug "NOTHING TO DO"
        return nothing, true
    elseif length(insert_locs) != length(circ.locs)
        @debug "INJECTION BY REPLACEMENT"
        @debug "RECURSIVE CALL REQUIRED"
        # Copy mapping and rearrange
        new_mapping = Dict()
        new_locs = []
        append!(new_locs, circ.locs...)

        inserted = false

        for (i, loc) in enumerate(circ.locs)
            if haskey(mapping, loc)
                # Find index of the location where circ.locs[j] == mapping[loc]
                # If it is not found, then we need to insert a location in circ.locs

                mapped_loc = findfirst(x -> x == mapping[loc], circ.locs)

                if isnothing(mapped_loc)
                    append!(new_locs, mapping[loc])
                    new_mapping[i] = length(new_locs)
                    inserted = true
                else
                    new_mapping[i] = mapped_loc
                end
            end
        end

        new_put = circ
        if inserted
            @debug "DIMENSION MISMATCH: RECOMPILATION REQUIRED"
            cpy = Yao.copy(Yao.subblocks(circ)[1])
            new_cpy, _ = remap(cpy, new_mapping, length(new_locs))
            new_put = Yao.control(new_ctrl_locs, new_locs => new_cpy)(expected_length)
        end
        return new_put, false
        #throw(ArgumentError("This module does not yet support partial remapping of the control block"))
    else
        if length(Yao.subblocks(circ)) != 1
            throw(ArgumentError("This module does not yet support multiple subblocks"))
        end
        
        return Yao.control(new_ctrl_locs, insert_locs => Yao.copy(Yao.subblocks(circ)[1]))(expected_length), true
    end

    return nothing, true
end

function remap(circ::Yao.CompositeBlock, mapping::Dict, expected_length::Int=Yao.nqubits(circ))::Tuple{Union{Union{Yao.AbstractBlock, Function}, Nothing}, Bool}
    @debug ""
    @debug "===== COMPOSITE ====="
    @debug circ

    block_modified = false

    if isa(circ, Yao.ChainBlock)
        if expected_length != Yao.nqubits(circ)
            old_blocks = Yao.subblocks(circ)
            circ = chain(expected_length)
            append!(Yao.subblocks(circ), old_blocks)
            block_modified = true
        end
    end

    subblocks = Yao.subblocks(circ)

    inserted = false

    for (i, subblock) in enumerate(subblocks)
        if inserted
            inserted = false
            continue
        end
        @debug "Remapping subblock $i"
        mblock, should_duplicate = remap(subblock, mapping, Yao.nqubits(circ))
        if !isnothing(mblock)
            if should_duplicate
                insert!(subblocks, i+1, mblock)
                inserted = true
            else
                subblocks[i] = mblock
            end
        end
    end

    if block_modified
        return circ, false
    end

    return nothing, true
end

function remap(circ::Yao.AbstractBlock, mapping::Dict, expected_length::Int=Yao.nqubits(circ))::Tuple{Union{Union{Yao.AbstractBlock, Function}, Nothing}, Bool}
    throw(ArgumentError("This module does not yet support this type of block: " * string(typeof(circ))))
end