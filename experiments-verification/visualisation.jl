using Plots
pyplot()


function plot_forecast(output, predictions, ix_trn, ix_val; tt=1, posterior=false)

    T = length(output)

    # Limits
    ylims = [minimum(output[ix_trn[1]:end]), maximum(output[ix_trn[1]:end])]

    # Plot training output
    plot(ix_trn[1:tt:end], output[ix_trn[1:tt:end]], color="blue", ylims=ylims, label="training data", size=(800,300))

    # Plot validation output
    scatter!(ix_val[1:tt:end], output[ix_val[1:tt:end]], color="black", ylims=ylims, label="validation data")

    # Plot fit and forecast
    if posterior
        sdev = sqrt.(predictions[2][ix_trn[1]:tt:T])
        plot!(ix_trn[1]:tt:T, predictions[1][ix_trn[1]:tt:T], ribbon=[sdev, sdev], color="purple", label="predictions")
    else
        plot!(ix_trn[1]:tt:T, predictions[ix_trn[1]:tt:T], color="purple", label="predictions")
    end
end


function plot_errors(output, predictions, ix_val)

    # Compute prediction errors
    pred_errors = (predictions[ix_val] - output[ix_val]).^2

    # Compute root mean square
    RMS = sqrt(mean(pred_errors))

    # Plot errors of time
    scatter(pred_errors, label="RMS = "*string(RMS), ylabel="squared error", yscale=:log10, size=(800,300))
end