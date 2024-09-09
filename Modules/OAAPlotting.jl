include("./OAACircuitBuilder.jl")

export plotmeasure;
export get_distribution;

using StatsBase: Histogram, fit
using Plots: bar, scatter!, gr; gr()
using BitBasis
using YaoPlots

# takes in the model and gets a sample distribution
# returns values of parameter bits
# TODO: specify number of samples
function learn_distribution(model::OAABlock)
	state = run_oaa(model)
	x = measure(state; nshots=100);

	# store the parameter lanes to be accessed
	param_lanes = Vector{Int64}()

	for i in 1:model.num_bits
		# measured lanes are:
		# 	rx rotation parameters
		# 	ry rotation parameters
		# 	cnot (bit transition) parameter
		m = model.architecture_list[i]["U"]
		append!(param_lanes, m.global_lane_map.rx_param_lanes)
		append!(param_lanes, m.global_lane_map.ry_param_lanes)
		if i != model.num_bits
			append!(param_lanes, [m.global_lane_map.cnot_param_lane])
		end
	end
	
	# create an empty vector to store param results
	measured_params = Vector{Vector{Int64}}()

	# iterate through and store parameter results
	for i in 1:length(x)
		push!(measured_params, x[i][param_lanes])
	end

	return measured_params
end

# measured_params = set of results
# rotation_precision = rotation precision (Int)
# b = number of bits in the data
function get_hypothesis(measured_params::Vector{Vector{Int64}}, rotation_precision::Int64, b::Int64)
	# for every sample
	results = Vector{DitStr{2, b, Int64}}();

	for s in 1:length(measured_params)
		sample = measured_params[s] 
		# for every bit in the data point
		# create an empty chain with number of bits
		circuit = chain(b)
		for j in 1:b
			for r in 1:rotation_precision
				# x param index = 2 * rotation_precision * (j - 1) + 2*(i - 1) + r
				# y param index = 2 * rotation_precision * (j - 1) + 2*(i - 1) + rotation_precision + r
				# angle = MAX_ROTATION / 2^(r)
				# apply X rotation
				if sample[2 * rotation_precision * (j - 1) + r] == 1
					# add Rx gate with rotatin MAX_ROTATION / 2^(r)
					circuit = chain(b,
						subroutine(circuit, 1:b), # append to circuit
						put(j => Rx(MAX_ROTATION / 2^(r))) # apply Rx rotation to jth bit
					);
				end
				if sample[2 * rotation_precision * (j - 1) + rotation_precision + r] == 1
					# add Ry gate with rotation MAX_ROTATION / 2^(r)
					circuit = chain(b,
						subroutine(circuit, 1:b), # append to circuit
						put(j => Ry(MAX_ROTATION / 2^(r))) # apply Rx rotation to jth bit
					);
				end
			end

			# apply CNOT gate if parameter bit == 1
			if j != b
				if sample[length(sample)] == 1
					circuit = chain(b, 
						subroutine(circuit, 1:b),
						cnot(j, j + 1)
					)
				end
			end
			
				
			if j != b
				# add CNOT gate
				circuit = chain(b, 
					subroutine(circuit, 1:b),
					cnot(j, j+1)
				);
			end
		end
		# measure data point
		state = zero_state(b);
		focus!(state, 1:b);
		state |> circuit
		relax!(state, 1:b);

		result = measure(state, nshots=100);
			
		# append to results
		append!(results, result)
	end

	# return results
	return results
end

# plots the results of specified array in a histogram
function plotmeasure(x::Vector{})
	# TODO: insert checks
	b = length(x[1])
	xInt = Int.(x)
	st = length(xInt)

	# n = skeleton total num lanes
	hist = fit(Histogram, xInt, 0:2^b)

	num_entries = length(hist.weights)
	sorted_indices = 1:length(xInt)
	colors = nothing
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

	bar(
		hist.edges[1][begin:num_entries],
		hist.weights[sorted_indices[begin:num_entries]],
		title = "Histogram", label="Found in "*string(st)*" tries",
		size=(600*(num_entries)/s,400),
		ylims=(0, maximum(hist.weights)),
		xlims=(-0.5, num_entries-0.5),
		grid=:false, ticks=false,
		border=:none,
		color=(isnothing(colors) ? (:lightblue) : colors[begin:num_entries]),
		lc=:lightblue,
		foreground_color_legend = nothing,
		background_color_legend = nothing
	)

	scatter!(0:num_entries-1, ones(num_entries,1), markersize=0, label=:none,
		series_annotations="|" .* string.(hist.edges[1][sorted_indices[begin:num_entries]]; base=2, pad=b) .* "⟩")
	scatter!(0:num_entries-1, zeros(num_entries,1) .+ maximum(hist.weights), markersize=0, label=:none, series_annotations=string.(hist.weights[sorted_indices[begin:num_entries]]))
end