module GroverPlotting

    using StatsBase: Histogram, fit
	using Plots: bar, scatter!, gr; gr()
	using BitBasis
    using YaoPlots

    export plotmeasure
    export configureYaoPlots

	function plotmeasure(x::Array{BitStr{n,Int},1}; oracle_function::Union{Function, Nothing} = nothing, sort=false, num_entries=nothing) where n
		xInt = Int.(x)
		st = length(xInt)
		hist = fit(Histogram, xInt, 0:2^n)
		sorted_indices = nothing
		if isnothing(num_entries)
			num_entries = length(hist.weights)
		end
		if sort
			sorted_indices = sortperm(hist.weights, rev=true)
		else
			sorted_indices = 1:length(xInt)
		end 
		colors = nothing
		if !isnothing(oracle_function)
			colors = [oracle_function(sorted_indices[i]) ? :red : :lightblue for i in 1:2^n]
		end
		x = 0
		if(n<=3)
			s=8
		elseif(n>3 && n<=6)
			s=5
		elseif(n>6 && n<=10)
			s=3.2
		elseif(n>10 && n<=15)
			s=2
		elseif(n>15)
			s=1
		end

		bar(hist.edges[1][begin:num_entries], hist.weights[sorted_indices[begin:num_entries]], title = "Histogram", label="Found in "*string(st)*" tries", size=(600*(num_entries)/s,400), ylims=(0, maximum(hist.weights)), xlims=(-0.5, num_entries-0.5), grid=:false, ticks=false, border=:none, color=(isnothing(colors) ? (:lightblue) : colors[begin:num_entries]), lc=:lightblue, foreground_color_legend = nothing, background_color_legend = nothing)
		scatter!(0:num_entries-1, ones(num_entries,1), markersize=0, label=:none,
			series_annotations="|" .* string.(hist.edges[1][sorted_indices[begin:num_entries]]; base=2, pad=n) .* "⟩")
		scatter!(0:num_entries-1, zeros(num_entries,1) .+ maximum(hist.weights), markersize=0, label=:none, series_annotations=string.(hist.weights[sorted_indices[begin:num_entries]]))

	end

    function configureYaoPlots()
        YaoPlots.CircuitStyles.linecolor[] = "black" 
        YaoPlots.CircuitStyles.gate_bgcolor[] = "white" 
        YaoPlots.CircuitStyles.textcolor[] = "#000000"
        YaoPlots.CircuitStyles.fontfamily[] = "JuliaMono"
        #YaoPlots.CircuitStyles.lw[] = 2.5
        #YaoPlots.CircuitStyles.textsize[] = 13
        #YaoPlots.CircuitStyles.paramtextsize[] = 8
    end
end