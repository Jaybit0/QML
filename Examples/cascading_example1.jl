# ======== IMPORTS ========
# =========================
include("../Modules/OAACircuitBuilder.jl")
if !isdefined(Main, :QML)
    include("../Modules/QML.jl")
    using .QML
end


using Yao
using Yao.EasyBuild, YaoPlots

# num_model_lanes = 2
rotation_precision = 2

training_data = [[0,0],[1,0]]
# training_data = [[0,0, 1],[1,0, 0], [1,1, 0]]

model = create_oaa_circuit(training_data, rotation_precision)

vizcircuit(model.architecture)

state = run_oaa(model)

println(results)
# REDEFINED PLOTMEASURE
x = measure(state; nshots=1000)

xInt = Int.(x)
st = length(xInt)

# n = skeleton total num lanes
using StatsBase: Histogram, fit
hist = fit(Histogram, xInt, 0:2^8)
println(typeof(hist))

Plots.plot(hist)

# TODO: convert x-axis values into bit strings

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

# bar(hist.edges[1][begin:num_entries], hist.weights[sorted_indices[begin:num_entries]])
# bar(
#     hist.edges[1][begin:num_entries],
#     hist.weights[sorted_indices[begin:num_entries]],
#     title = "Histogram", label="Found in "*string(st)*" tries",
#     size=(600*(num_entries)/s,400),
#     ylims=(0, maximum(hist.weights)),
#     xlims=(-0.5, num_entries-0.5),
#     grid=:false, ticks=false,
#     border=:none,
#     color=(isnothing(colors) ? (:lightblue) : colors[begin:num_entries]),
#     lc=:lightblue,
#     foreground_color_legend = nothing,
#     background_color_legend = nothing
# )

scatter!(0:num_entries-1, ones(num_entries,1), markersize=0, label=:none,
	series_annotations="|" .* string.(hist.edges[1][sorted_indices[begin:num_entries]]; base=2, pad=n) .* "⟩")
scatter!(0:num_entries-1, zeros(num_entries,1) .+ maximum(hist.weights), markersize=0, label=:none, series_annotations=string.(hist.weights[sorted_indices[begin:num_entries]]))


## -- ARCHIVED -- 

# skeleton = model

# models = skeleton.models
# transitions = skeleton.transition_models

# iter = skeleton.num_bits

# # set up initial state
# n = skeleton.total_num_lanes;

# state = zero_state(n);

# println("Initial state")
# println(state)

# # define R0lstar
# R0lstar = chain(
#     skeleton.num_bits + skeleton.rotation_precision + 1,
#     repeat(X, skeleton.num_bits + 2:skeleton.num_bits + skeleton.rotation_precision + 1),
#     cz(skeleton.num_bits + 2:skeleton.num_bits + skeleton.rotation_precision, skeleton.num_bits + skeleton.rotation_precision + 1),
#     repeat(X, skeleton.num_bits + 2:skeleton.num_bits + skeleton.rotation_precision + 1),
# );

# # R0lstar = chain(
# #     skeleton.rotation_precision + 1,
# #     repeat(X, 2:skeleton.rotation_precision + 1),
# #     cz(2:2:skeleton.rotation_precision, skeleton.rotation_precision + 1),
# #     repeat(X, 2:skeleton.rotation_precision + 1),
# # );


# for i in 1:iter
#     # organize lanes
#     target_lane = models[i].global_lane_map.target_lane;

#     ## get RxChain and lanes
#     collected_rx_lanes = vcat([target_lane], models[i].global_lane_map.rx_model_lanes, models[i].global_lane_map.rx_param_lanes);
#     # collected_rx_lanes = vcat([target_lane], models[i].global_lane_map.rx_param_lanes);

#     ## RyChain and lanes
#     collected_ry_lanes = vcat([target_lane], models[i].global_lane_map.ry_model_lanes, models[i].global_lane_map.ry_param_lanes);
#     # collected_ry_lanes = vcat([target_lane], models[i].global_lane_map.ry_param_lanes);

#     # run state through first model
#     ## focus Rx lanes
#     focus!(state, collected_rx_lanes);

#     ## pipe state into RxChain
#     state |> models[i].rx_compiled_architecture;
        
#     ## measure outcome
#     outcome = measure!(state, 1)

#     ## if outcome != 0, run OAA again
#     if outcome != 0
#         state |> Daggered(model[i].rx_compiled_architecture);
#         state |> R0lstar;
#         state |> model[i].rx_compiled_architecture;
#     end

#     ## relax Rx lanes
#     relax!(state, collected_rx_lanes);

#     ## focus Ry lanes
#     focus!(state, collected_ry_lanes);

#     ## pipe state into RyChain
#     state |> models[i].ry_compiled_architecture;

#     ## measure outcome
#      outcome = measure!(state, 1)

#     ## if outcome != 0, run OAA again
#     if outcome != 0
#         state |> Daggered(models[i].ry_compiled_architecture);
#         state |> R0lstar;
#         state |> models[i].ry_compiled_architecture;
#     end

#     ## relax Ry lanes
#     relax!(state, collected_ry_lanes);

#     # if i != iter
#         # append the transition model
#     if i != iter
#         transition_lane_map = compile_lane_map(transitions[i].global_lane_map)
#         focus!(state, transition_lane_map)
#         state |> transitions[i].architecture
#         relax!(state, transition_lane_map)
#     end
# end


