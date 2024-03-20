function remapCircuit(circ::Yao.AbstractBlock, mapping::Dict)::Union{Union{Yao.AbstractBlock, Function}, Nothing}
    if isa(circ, Yao.PutBlock)
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

        if length(insert_locs) == 0 
            return nothing
        elseif length(insert_locs) != length(circ.locs)
            throw(ArgumentError("This module does not yet support partial remapping"))
        else
            if length(Yao.subblocks(circ)) != 1
                throw(ArgumentError("This module does not yet support multiple subblocks"))
            end
            
            return Yao.put(insert_locs => Yao.copy(Yao.subblocks(circ)[1]))(Yao.nqubits(circ))
        end
    end

    if isa(circ, Yao.ControlBlock)
        return nothing
    end

    if isa(circ, Yao.CompositeBlock)
        subblocks = Yao.subblocks(circ)

        inserted = false

        for (i, subblock) in enumerate(subblocks)
            if inserted
                inserted = false
                continue
            end
            mblock = remapCircuit(subblock, mapping)
            if !isnothing(mblock)
                insert!(subblocks, i, mblock)
                inserted = true
            end
        end
    end
end