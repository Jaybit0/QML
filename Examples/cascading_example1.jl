# ======== IMPORTS ========
# =========================
if !isdefined(Main, :QML)
    include("../Modules/QML.jl")
    using .QML
end
include("../Modules/OAACircuitBuilder.jl")
include("../Modules/OAAPlotting.jl")

using Yao
using Yao.EasyBuild, YaoPlots
using Plots

using DataStructures


# num_model_lanes = 2
rotation_precision = 2;

training_data = [[1], [1], [1], [1]];

model = create_oaa_circuit(training_data, rotation_precision);

vizcircuit(model.architecture)

# TODO: fix register mismatch
measured_params = learn_distribution(model);
counter(measured_params)

n = length(training_data) # number data points
b = length(training_data[1]) # number bits

hypothesis = get_hypothesis(measured_params, rotation_precision, b);

plotmeasure(hypothesis)

vis = chain(model.total_num_lanes);

for i in 1:length(model.architecture_list)
	m = model.architecture_list[i]

	u_model = m["U"]
	cnot = m["CNOT"]

	target_lane = u_model.global_lane_map.target_lane;
	collected_rx_lanes = vcat([target_lane], u_model.global_lane_map.rx_model_lanes, u_model.global_lane_map.rx_param_lanes);
	collected_ry_lanes = vcat([target_lane], u_model.global_lane_map.ry_model_lanes, u_model.global_lane_map.ry_param_lanes);

	vis = chain(
		model.total_num_lanes,
		subroutine(vis, 1:model.total_num_lanes),
		subroutine(u_model.rx_compiled_architecture, collected_rx_lanes),
		subroutine(cnot.architecture, cnot.global_lane_map.lanes),
		subroutine(u_model.ry_compiled_architecture, collected_ry_lanes),
		subroutine(cnot.architecture, cnot.global_lane_map.lanes)
	)
	if i != model.num_bits
		t = m["TRANSITION"]
		t_lane_map = compile_lane_map(t)
		vis = chain(
			model.total_num_lanes,
			subroutine(vis, 1:model.total_num_lanes),
			subroutine(t.architecture, t_lane_map)
		)
	end
end

# ## -- START: test all-in-one results
# results = run_oaa(model)

# hypothesis_lanes = model.total_num_lanes + 1:model.total_num_lanes + b

# measured_hypothesis = Vector{Vector{Int64}}()

# for result in results
# 	push!(measured_hypothesis, result[hypothesis_lanes])
# end

# using Counters

# counter(measured_hypothesis)
# ## -- END: test all-in-one results


# ## -- START: visualizing distribution of target bits
arch_list = model.architecture_list
param_lanes = Vector{Vector{Int}}()

for i in 1:length(arch_list)
	t = arch_list[i]["U"]
	temp = Vector{Int}()
	append!(temp, t.global_lane_map.rx_param_lanes)
	append!(temp, t.global_lane_map.ry_param_lanes)

	if i != b
		push!(temp, t.global_lane_map.cnot_param_lane)
	end
	push!(param_lanes, temp)
end

results = Vector{UInt64}()

for i in 1:length(measured_params)
	push!(results, parse(UInt, join(string.(measured_params[i])); base=2))
end

# for result in measured_params
# 	push!(results, parse(UInt, join(string.(result[param_lanes]))))
# end

histogram(results)
# ## -- END: visualizing distribution of target bits


# ## -- START: visualizing the measured_params --
# temp = Vector{UInt64}()
# for i in 1:length(measured_params)
# 	push!(temp, parse(UInt, join(string.(measured_params[i])); base=2))
# end

# histogram(temp)

# ## -- END: visualizing the measured_params --

## -- START: plotmeasure troubleshooting -- ##
x = hypothesis


b = length(x[1])
xInt = Int.(x)
st = length(xInt)


using Plots
histogram(xInt)

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

## -- END: plotmeasure troubleshooting -- ##


# # -- START: viz oaa circuit --

# skeleton = model

# models = skeleton.models
# transitions = skeleton.transition_models

# R0lstar = chain(
#     skeleton.num_bits - 2 + 2 * skeleton.rotation_precision + 1,
#     repeat(X, skeleton.num_bits + 2:skeleton.num_bits + skeleton.rotation_precision + 1),
#     cz(skeleton.num_bits + 2:skeleton.num_bits + skeleton.rotation_precision, skeleton.num_bits + skeleton.rotation_precision + 1),
#     repeat(X, skeleton.num_bits + 2:skeleton.num_bits + skeleton.rotation_precision + 1),
# );

# lanes = vcat()

# n = skeleton.total_num_lanes;

# vizcircuit(R0lstar)

# temp = chain(n,
#     subroutine(R0lstar, 1:n),
#     subroutine(Daggered(models[1].rx_compiled_architecture), 1:n),
#     subroutine(R0lstar, 1:n),
#     subroutine(models[1].rx_compiled_architecture, 1:n)
# )
# # -- END: viz oaa circuit