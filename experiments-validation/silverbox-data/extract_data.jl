using DataFrames
using CSV
using JLD
using Plots
pyplot()

"""Data"""

# Read data from CSV file
df = CSV.read(pwd()*"/data/SNLS80mV.csv", ignoreemptylines=true)
df = select(df, [:V1, :V2])

# Number of samples
num_samples = size(df, 1)

"""Visualization"""

# Check full signals
scatter(1:num_samples, df[:,1], markerstrokewidth=0, markersize=1, size=(1000,400))
xlabel!("samples")
title!("input")
savefig(pwd()*"/data/input.png")

scatter(1:num_samples, df[:,2], markerstrokewidth=0, markersize=1, size=(1000,400))
xlabel!("samples")
title!("output")
savefig(pwd()*"/data/output.png")

# Zooms
zoom = 1:100;

# Check full signals
plot(zoom, df[zoom,1], markerstrokewidth=0, markersize=1, size=(1000,400))
xlabel!("samples")
title!("input")
savefig(pwd()*"/data/input_zoom.png")

plot(zoom, df[zoom,2], markerstrokewidth=0, markersize=1, size=(1000,400))
xlabel!("samples")
title!("output")
savefig(pwd()*"/data/output_zoom.png")

# """Saving"""
#
# #Save the data-frame
# fid = jldopen(pwd()*"/data/SNLS80mV.jld", "w");
# write(fid, "Silverbox", df);
# close(fid);
