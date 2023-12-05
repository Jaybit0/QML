module GroverPlotting

    using StatsBase: Histogram, fit
	using Plots: bar, scatter!, gr; gr()
	using BitBasis
    using YaoPlots

    export plotmeasure
    export configureYaoPlots

	function plotmeasure(x::Array{BitStr{n,Int},1}, st="#") where n
		hist = fit(Histogram, Int.(x), 0:2^n)
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
		bar(hist.edges[1] .- 0.5, hist.weights, title = "Histogram", label="Found in "*string(st)*" tries", size=(600*(2^n)/s,400), ylims=(0, maximum(hist.weights)), xlims=(0, 2^n), grid=:false, ticks=false, border=:none, color=:lightblue, lc=:lightblue, foreground_color_legend = nothing, background_color_legend = nothing)
		scatter!(0:2^n-1, ones(2^n,1), markersize=0, label=:none,
		series_annotations="|" .* string.(hist.edges[1]; base=2, pad=n) .* "⟩")
		scatter!(0:2^n-1, zeros(2^n,1) .+ maximum(hist.weights), markersize=0, label=:none, series_annotations=string.(hist.weights))
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