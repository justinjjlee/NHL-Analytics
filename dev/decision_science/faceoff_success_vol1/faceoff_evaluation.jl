using Pkg
using DataFrames, CSV
using DBInterface, MySQL
using Plots, StatsPlots, LaTeXStrings

cd(@__DIR__)
str_cd_df = # Your file location
serverloc = # Your file location
serverfolder = "nhl/data_gamelvl/processed"
str_cd_df = serverloc * serverfolder
# Pull data
df_seq = CSV.read(str_cd_df*"/2022_02_possession_sequence.csv", DataFrame)

# Filter out missing value
df_seq = filter(:zone_faceoff_home => x -> !(ismissing(x) || isnothing(x)), df_seq)

# Plot some conditional probabilities
# First - remove neutral zone start
df_seq = df_seq[df_seq.zone_faceoff_home .!= "center", :]

groupfaceoff = groupby(df_seq, [:posture, :faceoff_won_home, :goal, :zone_change])

# Disaggregated count 
group_tab = combine(groupfaceoff, :gameIdx=>length)

# Count subtotal ..................................

# Base of conditional probabilities, and join
subtotal_group = groupby(df_seq, [:posture])
subtotal_0 = combine(subtotal_group, :gameIdx=>length=>:subtotal_posture)
group_tab = innerjoin(group_tab, subtotal_0, on=:posture)

subtotal_group = groupby(df_seq, [:faceoff_won_home])
subtotal_0 = combine(subtotal_group, :gameIdx=>length=>:subtotal_faceoff_won_home)
group_tab = innerjoin(group_tab, subtotal_0, on=:faceoff_won_home)
# Include posture/utility status
subtotal_group = groupby(df_seq, [:posture, :faceoff_won_home])
subtotal_0 = combine(subtotal_group, :gameIdx=>length=>:subtotal_both)
group_tab = innerjoin(group_tab, subtotal_0, on=[:posture, :faceoff_won_home])
# Include zone change
subtotal_group = groupby(df_seq, [:posture, :faceoff_won_home, :zone_change])
subtotal_0 = combine(subtotal_group, :gameIdx=>length=>:subtotal_three)
group_tab = innerjoin(group_tab, subtotal_0, on=[:posture, :faceoff_won_home, :zone_change])


# Calculate P(score | posture)
pcon_0 = combine(groupby(group_tab, [:posture, :goal]), 
    [
        :gameIdx_length=>sum=>:count, 
        :subtotal_posture=>last=>:total
    ]
)
# Start with count total and calculate probability
pcon_0_count = unstack(pcon_0, [:goal], :posture, :count)
pcon_0_total = unstack(pcon_0, [:goal], :posture, :total)
pcon_0_ratio_season = pcon_0_count ./ pcon_0_total
pcon_0_ratio_season[:,:goal] = pcon_0_count[:,:goal]
pcon_0_ratio_season

# Calculate P(score | faceoff)
pcon_1 = combine(groupby(group_tab, [:faceoff_won_home, :goal]), 
    [
        :gameIdx_length=>sum=>:count, 
        :subtotal_faceoff_won_home=>last=>:total
    ]
)
# Start with count total and calculate probability
pcon_1_count = unstack(pcon_1, [:goal], :faceoff_won_home, :count)
pcon_1_total = unstack(pcon_1, [:goal], :faceoff_won_home, :total)
pcon_1_ratio_season = pcon_1_count ./ pcon_1_total
pcon_1_ratio_season[:,:goal] = pcon_1_count[:,:goal]
pcon_1_ratio_season

# Calculate P(score | posture, faceoff)
pcon_2 = combine(groupby(group_tab, [:faceoff_won_home, :posture, :goal]), 
    [
        :gameIdx_length=>sum=>:count, 
        :subtotal_both=>last=>:total
    ]
)
pcon_2_count = unstack(pcon_2, [:faceoff_won_home, :posture], :goal, :count)
pcon_2_total = unstack(pcon_2, [:faceoff_won_home, :posture], :goal, :total)
pcon_2_ratio_season = pcon_2_total[:,[:faceoff_won_home, :posture]]
pcon_2_ratio_season[:,:goal_against] = pcon_2_count[:,"-1"]./pcon_2_total[:,"-1"]
pcon_2_ratio_season[:,:goal_for] = pcon_2_count[:,"1"]./pcon_2_total[:,"1"]
pcon_2_ratio_season

# Validation: Calculate P(score | posture, faceoff, zone change)
pcon_3 = combine(groupby(group_tab, [:faceoff_won_home, :posture, :goal, :zone_change]), 
    [
        :gameIdx_length=>sum=>:count, 
        :subtotal_three=>last=>:total
    ]
)
pcon_3_count = unstack(pcon_3, [:faceoff_won_home, :posture, :zone_change], :goal, :count)
pcon_3_total = unstack(pcon_3, [:faceoff_won_home, :posture, :zone_change], :goal, :total)
pcon_3_ratio_season = pcon_3_total[:,[:faceoff_won_home, :posture, :zone_change]]
pcon_3_ratio_season[:,:goal_against] = pcon_3_count[:,"-1"]./pcon_3_total[:,"-1"]
pcon_3_ratio_season[:,:goal_for] = pcon_3_count[:,"1"]./pcon_3_total[:,"1"]
pcon_3_ratio_season


# Plotting ..............................................................
# P(score | posture)
colorscheme_1 = ["mediumaquamarine", "tomato","honeydew4","mediumaquamarine"]
templot = reshape(Matrix(pcon_1_ratio_season[[3,1],2:end])',4,1)
xlabels_1 = [
    "Faceoff lost\nGoal scored for",
    "Faceoff won\nGoal scored for",
    "Faceoff lost\nGoal scored against",
    "Faceoff won\nGoal scored against",
    ]
pltsave = bar(xlabels_1, templot, 
    #yerr=[0.02,0.02,0.02,0.02],
    title="Goal-Scoring Probabililty by Faceoff Outcomes", 
    fill=colorscheme_1,
    size=(760,550), legend=false, show=true, dpi=300
)
savefig(pltsave, "condprob.png")

# Plot P(score | posture, faceoff)
xlabels = [
    "Faceoff lost\nDefense",
    "Faceoff lost\nOffense",
    "Faceoff won\nDefense",
    "Faceoff won\nOffense",
    ]
colorscheme_2_1 = ["tomato", "mediumaquamarine","mediumaquamarine","honeydew4"]
colorscheme_2_2 = ["mediumaquamarine", "honeydew4","tomato","mediumaquamarine"]
pltsave = plot(
    bar(xlabels, pcon_2_ratio_season[:, :goal_for], title="Goal Scored For", fill=colorscheme_2_1),
    bar(xlabels, pcon_2_ratio_season[:, :goal_against], title="Goal Scored Against", fill=colorscheme_2_2),
    size=(760,550), 
    legend=false, dpi=300
)
savefig(pltsave, "condprob_detail.png")


#...........................................................................
# calculate the statistical signifcance difference in probability
#   Using the bootstrap method
n_boot = 10_000
n_sample = 5_000
L = 1:size(df_seq,1) # data list to draw from 
α = 0.1 # Error rate

# Initialize bucket
pcon_0_3d = zeros(3,2,n_boot)
pcon_1_3d = zeros(3,2,n_boot)
pcon_2_3d = zeros(4,2,n_boot)
# Bootstrap computation
for i_boot in 1:n_boot
    # Random select rows
    i_rows = rand(L, n_sample)
    df_sample = df_seq[i_rows, :]

    # Repeat above process
    groupfaceoff = groupby(df_sample, [:posture, :faceoff_won_home, :goal])

    # Disaggregated count 
    group_tab = combine(groupfaceoff, :gameIdx=>length)

    # Count subtotal ..................................

    # Base of conditional probabilities, and join
    subtotal_group = groupby(df_sample, [:posture])
    subtotal_0 = combine(subtotal_group, :gameIdx=>length=>:subtotal_posture)
    group_tab = innerjoin(group_tab, subtotal_0, on=:posture)

    subtotal_group = groupby(df_sample, [:faceoff_won_home])
    subtotal_0 = combine(subtotal_group, :gameIdx=>length=>:subtotal_faceoff_won_home)
    group_tab = innerjoin(group_tab, subtotal_0, on=:faceoff_won_home)

    subtotal_group = groupby(df_sample, [:posture, :faceoff_won_home])
    subtotal_0 = combine(subtotal_group, :gameIdx=>length=>:subtotal_both)
    group_tab = innerjoin(group_tab, subtotal_0, on=[:posture, :faceoff_won_home])

    # Calculate P(score | posture)
    pcon_0 = combine(groupby(group_tab, [:posture, :goal]), 
        [
            :gameIdx_length=>sum=>:count, 
            :subtotal_posture=>last=>:total
        ]
    )
    # Start with count total and calculate probability
    pcon_0_count = unstack(pcon_0, [:goal], :posture, :count)
    pcon_0_total = unstack(pcon_0, [:goal], :posture, :total)
    pcon_0_ratio = pcon_0_count ./ pcon_0_total
    pcon_0_ratio[:,:goal] = pcon_0_count[:,:goal]
    pcon_0_3d[:,:,i_boot] .= pcon_0_ratio[:, 2:end]

    # Calculate P(score | faceoff)
    pcon_1 = combine(groupby(group_tab, [:faceoff_won_home, :goal]), 
        [
            :gameIdx_length=>sum=>:count, 
            :subtotal_faceoff_won_home=>last=>:total
        ]
    )
    # Start with count total and calculate probability
    pcon_1_count = unstack(pcon_1, [:goal], :faceoff_won_home, :count)
    pcon_1_total = unstack(pcon_1, [:goal], :faceoff_won_home, :total)
    pcon_1_ratio = pcon_1_count ./ pcon_1_total
    pcon_1_ratio[:,:goal] = pcon_1_count[:,:goal]
    pcon_1_3d[:,:,i_boot] .= pcon_1_ratio[:, 2:end]


    # Calculate P(score | posture, faceoff)
    pcon_2 = combine(groupby(group_tab, [:faceoff_won_home, :posture, :goal]), 
        [
            :gameIdx_length=>sum=>:count, 
            :subtotal_posture=>last=>:total
        ]
    )
    pcon_2_count = unstack(pcon_2, [:faceoff_won_home, :posture], :goal, :count)
    pcon_2_total = unstack(pcon_2, [:faceoff_won_home, :posture], :goal, :total)
    pcon_2_ratio = pcon_2_total[:,[:faceoff_won_home, :posture]]
    pcon_2_ratio[:,:goal_against] = pcon_2_count[:,"-1"]./pcon_2_total[:,"-1"]
    pcon_2_ratio[:,:goal_for] = pcon_2_count[:,"1"]./pcon_2_total[:,"1"]
    pcon_2_3d[:,:,i_boot] .= pcon_2_ratio[:, 3:end]
end

#...........................................................................
# Plot Density P(score | posture)
pltsave = density(pcon_1_3d[3,1,:], color=colorscheme_1[1], 
    linestyle=:dash,width=3,label=xlabels_1[1],
    legend=:topright,
    size=(760,550), dpi=300
)
density!(pcon_1_3d[3,2,:], color=colorscheme_1[2],width=9,label=xlabels_1[2])
density!(pcon_1_3d[1,1,:], color=colorscheme_1[3],width=9,label=xlabels_1[3])
density!(pcon_1_3d[1,2,:], color=colorscheme_1[4],width=3,label=xlabels_1[4])
title!("Conditional Probability of Goal Scored")
xlabel!("Goal-score probability "*L"p")
plot!([pcon_1_ratio_season[1,3]], seriestype="vline", label=false, color="black", linestyle=:dash)
plot!([pcon_1_ratio_season[3,3]], seriestype="vline", label=false, color="black", linestyle=:dash)

savefig(pltsave, "condprob_dist.png")

# Plot P(score | posture, faceoff)
p1 = density(pcon_2_3d[1,2,:], label=xlabels[1],width=9, color=colorscheme_2_1[1],
    title="Goal Scored For"
)
p1 = density!(pcon_2_3d[2,2,:], label=xlabels[2],width=2, linestyle=:dash, color=colorscheme_2_1[2])
p1 = density!(pcon_2_3d[3,2,:], label=xlabels[3],width=3, color=colorscheme_2_1[3])
p1 = density!(pcon_2_3d[4,2,:], label=xlabels[4],width=9, color=colorscheme_2_1[4])
p1 = plot!([pcon_2_ratio_season[2,4]], seriestype="vline", label=false, color="black", linestyle=:dash)
p1 = plot!([pcon_2_ratio_season[4,4]], seriestype="vline", label=false, color="black", linestyle=:dash)
p1 = xlabel!("Goal-score probability "*L"p")

p2 = density(pcon_2_3d[1,1,:], label=xlabels[1],width=2, color=colorscheme_2_2[1],
    title="Goal Scored Against", linestyle=:dash
)
p2 = density!(pcon_2_3d[2,1,:], label=xlabels[2],width=9, color=colorscheme_2_2[2])
p2 = density!(pcon_2_3d[3,1,:], label=xlabels[3],width=9, color=colorscheme_2_2[3])
p2 = density!(pcon_2_3d[4,1,:], label=xlabels[4],width=3, color=colorscheme_2_2[4])
p2 = plot!([pcon_2_ratio_season[1,3]], seriestype="vline", label=false, color="black", linestyle=:dash)
p2 = plot!([pcon_2_ratio_season[2,3]], seriestype="vline", label=false, color="black", linestyle=:dash)
p2 = xlabel!("Goal-score probability "*L"p")
pltsave = plot(p1,p2,
    legend=:topright,
    size=(760,550), dpi=300
)
savefig(pltsave, "condprob_detail_dist.png")

#...........................................................................
# Appendix: Calculate quantile of the draws, save as DataFrame
function quantile_3d(df, pct)
    # Quantile calculation of 3-d matrix
    res_mat = zeros(size(df,1), size(df,2))
    for irow in 1:size(df,1)
        for icol in 1:size(df,2)
            res_mat[irow, icol] = quantile(df[irow,icol,:], pct)
        end
    end
    return res_mat
end

pcon_0_ratio_boot_lower = copy(pcon_0_ratio)
pcon_0_ratio_boot_lower[:,2:end] .= quantile_3d( .- pcon_0_ratio, α/2)
pcon_0_ratio_boot_upper = copy(pcon_0_ratio)
pcon_0_ratio_boot_upper[:,2:end] .= quantile_3d(pcon_0_3d, 1-α/2)

pcon_1_ratio_boot_lower = copy(pcon_1_ratio)
pcon_1_ratio_boot_lower[:,2:end] .= quantile_3d(pcon_1_3d, α/2)
pcon_1_ratio_boot_upper = copy(pcon_1_ratio)
pcon_1_ratio_boot_upper[:,2:end] .= quantile_3d(pcon_1_3d, 1-α/2)