include("./YaoSupport.jl")

using StatsBase: Histogram, fit
using Plots: bar, scatter!, gr;
using Plots.PlotMeasures;
gr();
using BitBasis
using YaoPlots

export plotmeasure
export configureYaoPlots

function plotmeasure(grover_block::QMLBlock; sort=false, num_entries=nothing, split_annotations=false)
    register = zero_state(Yao.nqubits(grover_block))
	measured = register |> grover_block |> r->measure(r; nshots=100000)

	# plotmeasure(measured; oracle_function=grover_block.compiled_circuit.oracle_function, sort=sort, num_entries=num_entries)

	plotmeasure(measured; oracle_function=grover_block.compiled_circuit.oracle_function, sort=sort, num_entries=num_entries, grover_block=grover_block)
end

# duplicate function - temp for plotting purposes
function plotmeasure(x::Array{BitStr{n,Int},1}; oracle_function::Union{Function,Nothing}=nothing,
	sort=false,
	num_entries=nothing,
	grover_block=nothing
	) where {n}
    xInt = Int.(x)
    st = length(xInt)
    hist = fit(Histogram, xInt, 0:2^n)
    sorted_indices = nothing

    if isnothing(num_entries)
        num_entries = length(hist.weights)
    end
    num_entries = min(num_entries, length(hist.weights))
    if sort
        sorted_indices = sortperm(hist.weights, rev=true)
    else
        sorted_indices = 1:length(xInt)
	end

	if isnothing(grover_block)
		throw(ArgumentError("Please enter non-null QMLBlock grover block."))
	end

    colors = nothing
    if !isnothing(oracle_function)
        colors = [oracle_function(sorted_indices[i]) ? :red : :lightblue for i in 1:2^n]
    end
    x = 0
    if (n <= 3)
        s = 8
    elseif (n > 3 && n <= 6)
        s = 5
    elseif (n > 6 && n <= 10)
        s = 3.2
    elseif (n > 10 && n <= 15)
        s = 2
    elseif (n > 15)
        s = 1
    end

	tl = grover_block.compiled_circuit.target_lane

	formatted_annotations = format_kets(string.(hist.edges[1][sorted_indices[begin:num_entries]]; base=2, pad=n), tl, n)

    bar(
		hist.edges[1][begin:num_entries],
		hist.weights[sorted_indices[begin:num_entries]],
		title="Histogram",
		label="Found in " * string(st) * " tries",
		size=(600 * (num_entries) / s, 400),
		ylims=(0, maximum(hist.weights)),
		xlims=(-0.5, num_entries - 0.5),
		bar_width=0.75,
		bottom_margin=20mm,
		grid=:false, ticks=false, border=:none,
		color=(isnothing(colors) ? (:lightblue) : colors[begin:num_entries]),
		lc=:lightblue, foreground_color_legend=nothing, background_color_legend=nothing
		)
	
	scatter!(
		0:num_entries-1,
		ones(num_entries, 1),
		markersize=0,
		label=:none,
        series_annotations=formatted_annotations
	)
    scatter!(
		0:num_entries-1,
		zeros(num_entries, 1) .+ maximum(hist.weights),
		markersize=0,
		label=:none,
		series_annotations=string.(hist.weights[sorted_indices[begin:num_entries]])
	)
end

function format_kets(entries::Array{String}, target_index::Int, n::Int)
	# "|" .* string.(hist.edges[1][sorted_indices[begin:num_entries]]; base=2, pad=n) .* "⟩"
	if isnothing(entries)
		throw(ArgumentError("Please enter non-null string to format."))
	end

	n = length(entries)
	
	out = Array{String, 1}(undef, n)

	for i in 1:n
		val = entries[i]

		model = SubString(val, 1, target_index-1)
		target = val[target_index]
		param = SubString(val, target_index+1)

		formatted = string("Model: |", model, "⟩\nTarget: |", target, "⟩\nParameter: |", param, "⟩")
		# println(formatted)
		out[i] = formatted
	end

	return out
end

# function plotmeasure(x::Array{BitStr{n,Int},1}; oracle_function::Union{Function,Nothing}=nothing,
# 	sort=false,
# 	num_entries=nothing
# 	) where {n}
#     xInt = Int.(x)
#     st = length(xInt)
#     hist = fit(Histogram, xInt, 0:2^n)
#     sorted_indices = nothing
#     if isnothing(num_entries)
#         num_entries = length(hist.weights)
#     end
#     num_entries = min(num_entries, length(hist.weights))
#     if sort
#         sorted_indices = sortperm(hist.weights, rev=true)
#     else
#         sorted_indices = 1:length(xInt)
#     end
#     colors = nothing
#     if !isnothing(oracle_function)
#         colors = [oracle_function(sorted_indices[i]) ? :red : :lightblue for i in 1:2^n]
#     end
#     x = 0
#     if (n <= 3)
#         s = 8
#     elseif (n > 3 && n <= 6)
#         s = 5
#     elseif (n > 6 && n <= 10)
#         s = 3.2
#     elseif (n > 10 && n <= 15)
#         s = 2
#     elseif (n > 15)
#         s = 1
#     end

#     bar(
# 		hist.edges[1][begin:num_entries],
# 		hist.weights[sorted_indices[begin:num_entries]],
# 		title="Histogram",
# 		label="Found in " * string(st) * " tries",
# 		size=(600 * (num_entries) / s, 400),
# 		ylims=(0, maximum(hist.weights)),
# 		xlims=(-0.5, num_entries - 0.5),
# 		grid=:false, ticks=false, border=:none,
# 		color=(isnothing(colors) ? (:lightblue) : colors[begin:num_entries]),
# 		lc=:lightblue, foreground_color_legend=nothing, background_color_legend=nothing
# 		)
#     scatter!(
# 		0:num_entries-1,
# 		ones(num_entries, 1),
# 		markersize=0,
# 		label=:none,
#         series_annotations="|" .* string.(hist.edges[1][sorted_indices[begin:num_entries]]; base=2, pad=n) .* "⟩"
# 	)
#     scatter!(
# 		0:num_entries-1,
# 		zeros(num_entries, 1) .+ maximum(hist.weights),
# 		markersize=0,
# 		label=:none,
# 		series_annotations=string.(hist.weights[sorted_indices[begin:num_entries]])
# 	)
# end

function configureYaoPlots()
    YaoPlots.CircuitStyles.linecolor[] = "black"
    YaoPlots.CircuitStyles.gate_bgcolor[] = "white"
    YaoPlots.CircuitStyles.textcolor[] = "#000000"
    YaoPlots.CircuitStyles.fontfamily[] = "JuliaMono"
    #YaoPlots.CircuitStyles.lw[] = 2.5
    #YaoPlots.CircuitStyles.textsize[] = 13
    #YaoPlots.CircuitStyles.paramtextsize[] = 8
end

configureYaoPlots()