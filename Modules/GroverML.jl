module GroverML

    using Yao
    using Yao.EasyBuild, YaoPlots

    export prepareOracleGate
    export createGroverCircuit
    export createGroverIteration
    export computeCumProb
    export computeAngle
    export computePostGroverLikelihood
    export computeOptimalGroverN

    function prepareOracleGate(j::Integer, V::Yao.YaoAPI.AbstractBlock)
        return prepareOracleGate(j, V, Daggered(V))
    end

    function prepareOracleGate(j::Integer, V::Yao.YaoAPI.AbstractBlock, VInv::Yao.YaoAPI.AbstractBlock)::Yao.ChainBlock
        return chain(j, put(1:j => VInv), repeat(X, [1 : j]), control(1:j-1, j => Z), repeat(X, [1 : j]), put(1:j => V))
    end

    function createGroverCircuit(j::Integer, num_grover_iterations::Integer, groverIteration::Yao.YaoAPI.AbstractBlock)::Yao.ChainBlock
        totalCircuit = chain(j)

        for _ in 1:num_grover_iterations
            totalCircuit = chain(j, put(1:j => totalCircuit), put(1:j => groverIteration))
        end

        return totalCircuit
    end

    function createGroverIteration(j::Integer, diffusionGate::Yao.YaoAPI.AbstractBlock, oracle::Yao.YaoAPI.AbstractBlock)::Yao.ChainBlock
        return chain(j, put(1:j => diffusionGate), put(1:j => oracle))
    end

    function computeCumProb(reg::Yao.AbstractArrayReg, oracle::Function)::Float64
        result = 0
        p = probs(reg)

        for i in eachindex(p)
            if oracle(i)
                result += p[i]
            end
        end

        return result
    end

    function computeAngle(cumProb::Number)::Number
        return asin(sqrt(cumProb))
    end

    function computePostGroverLikelihood(angle::Number, nGrover::Integer)::Number
        return sin(angle + 2 * nGrover * angle)^2
    end

    function computeOptimalGroverN(cumProb::Number, thresh::Number = .9)::Integer
        angle = computeAngle(cumProb)
        optimum = 1/2 * (pi/(2 * angle) - 1)
        optimalN = trunc(Int, round(optimum))
        
        for i in 1:100
            l = computePostGroverLikelihood(angle, optimalN)
            if l >= thresh
                break
            end

            optimalN += 1

            if i >= 99
                throw(DomainError(cumProb, "Cannot find optimum using the threshold " * string(thresh)))
            end
        end

        return optimalN
    end

end