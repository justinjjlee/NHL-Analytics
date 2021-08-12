using Pkg

using Distributed
addprocs(8)
@everywhere using CSV, Glob, DataFrames

@everywhere str_cd = "C:\\Users\\justi\\Documents\\GitHub\\NHL-Analytics\\dev\\pythagorian_expectations\\"
@everywhere str_cd_dfs = "C:\\Users\\justi\\Google Drive\\Learning\\sports\\nhl\\backup_copy\\team_season\\"

# Pull all Data


function data_pull_merge(str_wd)
    files = glob("*.csv", str_wd)


    dfs = DataFrame.(CSV.File.(files))
    df = dfs[1]
    # add an index column to be able to later discern the different sources
    for i in 2:length(dfs)
        df = vcat(df, dfs[i], cols = :union)
    end
    # vcat requires additional parameter, it would not work with dataframe with different columns
    #df = reduce(vcat(), dfs)
    return df
end
df = data_pull_merge(str_cd_dfs)
df[:, :Season] = string.(df[:, :Season]); # Need to edit portion of dates attached
df[(length.(df.Season) .> 7), :Season] = [value[1:7] for (iter, value) in enumerate(df[(length.(df.Season) .> 7), :Season])]
# Pull necessary columns
df_fin = df[:, ["Team", "Season", "season_type", "Points", "Wins","Losses","Goals For", "Goals Against", "Goals For Per Game Played", "Goals Against Per Game Played"]]
sort!(df_fin, [:Season, :Team])
# Remove duplicates ... for some reason
df_fin = unique(df_fin)
df_fin[:, "Season Start"] = parse.(Int, [value[1:4] for (iter, value) in enumerate(df_fin[:, :Season])])

# For now, exclude playoff period 
df_fin_playoff = df_fin[(df_fin.season_type .== "playoff"), :]
df_fin = df_fin[(df_fin.season_type .== "regular"), :];

df_fin[:, "Wins Ratio"] = df_fin.Wins ./ (df_fin.Wins .+ df_fin.Losses)

df_fin[:, "Pythagorean Expectation"] = (df_fin."Goals For".^2) ./ ((df_fin."Goals For".^2) .+ (df_fin."Goals Against".^2))

# Identify stanley cup winner 
temp_df = combine(groupby(df_fin_playoff, [:Season]), :Wins => maximum => :Wins)
temp_df[:, "Stanley Cup"] = "Stanley Cup Winner";
df_fin_playoff = leftjoin(df_fin_playoff, temp_df, on = [:Season, :Wins])
df_fin_playoff[ismissing.(df_fin_playoff[:, "Stanley Cup"]), "Stanley Cup"] = "Lost in playoff";

# Identify playoff eligible team
df_fin = leftjoin(df_fin, df_fin_playoff[:, ["Team", "Season", "Stanley Cup"]], on = [:Team, :Season, ])
df_fin[ismissing.(df_fin[:, "Stanley Cup"]), "Stanley Cup"] = "Missed playoff";

cor(Array(df_fin[:, ["Wins", "Wins Ratio", "Pythagorean Expectation"]]))
# Historically, quite good expectation formation.

df_fin |> CSV.write(str_cd * "data_pythagorean_expectation.csv")