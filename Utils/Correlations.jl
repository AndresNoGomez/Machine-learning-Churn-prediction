# Assisting fuctions for the app 4 correlations study

# Calculate the correlations between the features and the target variable
function calculate_correlations_with_churn(dataset_df::DataFrame)

    dataset = copy(dataset_df)
    
    cols = names(dataset)
    correlations = []

    for col_name in cols
        correlacion = cor(dataset[:, Symbol("Churn Value")], dataset[:, Symbol(col_name)])
        push!(correlations, (col_name, correlacion))
    end

    cor_df = DataFrame(correlations, [:Variable, :Correlacion]);

    return cor_df

end


# Find the features with a correlation with the target variable smaller than a threshold and plot the values
function find_weak_correlations_with_churn(cor_df::DataFrame, threshold::Float64)

    removed_columns = []    
    labels = String[]

    for row in eachrow(cor_df)
        abs_cor = abs(row[:Correlacion])
        if abs_cor < threshold
            push!(removed_columns, Symbol(row[:Variable]))  
            # UNCOMMENT TO SEE THE REMOVED FEATURES AND THEIR CORRELATION WITH CHURN VALUE
            #println(row[:Variable], " with a correlation with Churn Value of: ", round(row[:Correlacion], digits=2))
            push!(labels, "")  
        else
            push!(labels, row[:Variable])
        end
    end
    println()
    println(length(removed_columns), " columns will be removed from the dataset.\n\n")


    default(size=(1600, 1300)) 
    bar_plot = bar(cor_df.Variable, cor_df.Correlacion,
    title = "Correlation between features and Churn Value",
    legend = false,
    xticks = (1:length(cor_df.Variable), labels),  
    xtickfontsize = 10, 
    rotation = 45,
    seriescolor = :green)   

    savefig(bar_plot, "images/correlation.png") 

    return removed_columns

end




# Find the features with a correlation greater than a threshold and plot the heatmap of the correlation matrix
function find_strong_correlations_among_features(cor_matrix::Matrix{Float64}, variable_names::Vector{String}, threshold::Float64)
    strong_correlations = []

    for i in 1:size(cor_matrix, 1)
        for j in i+1:size(cor_matrix, 2)
            if abs(cor_matrix[i, j]) > threshold
                push!(strong_correlations, (variable_names[i], variable_names[j], cor_matrix[i, j]))
            end
        end
    end

    heatmap_corr = heatmap(cor_matrix,
    title = "Correlations Matrix",
    xticks = (1:size(cor_matrix, 1), variable_names),
    yticks = (1:size(cor_matrix, 2), variable_names),
    color = :viridis,
    fontsize = 12,
    xrotation = 45)

    savefig(heatmap_corr, "images/heatmap.png")  
    return strong_correlations
end
