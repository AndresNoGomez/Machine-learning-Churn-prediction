# APPROACH4: Selection of the most relevant features for the churn rate.

# Numerical and binary variables are treated the same way as the other approaches.
# Zip code and city are encoded with their churn propotions.
# Al the other categorical categories are transformed by one-hot encoding to binary variables.
# For example, Contract wil be splitted in Contract_Month-to-Month, Contract_One-Year, Contract_Two-year.

# Then, we study the correlation between each variable and the Churn Value. We pick the ones with an absolute
# value greater than 0.2.
# Then, we study the correlation among the remaining variables to avoid multicollinearity.

function approach4(dataset_df::DataFrame)

    dataset = copy(dataset_df)
    expected_columns = 42  
    if size(dataset, 2) != expected_columns
        throw(ArgumentError("The DataFrame must have $expected_columns columns, but it has $(size(dataset, 2))."))
    end

    # Process binary and continuous columns
    process_binary_continuous_columns!(dataset)

    # Process the cyclical features: latitude and longitude.
    process_latitude_and_longitude!(dataset)

    # Encode City and Zip Value according to the Churn rates:
    categorical_columns_to_transform = ["City", "Zip Code"]
    propotion_encoder!(categorical_columns_to_transform, dataset)
    
    # Encode the other categoricals as One-Hot and split them as binary variables.
    categorical_columns = ["Offer", "Internet Type", "Contract", "Payment Method"]
    onehot_columns = []

    for col_name in categorical_columns
        column_data = dataset[:, Symbol(col_name)]
        classes = unique(column_data)
        onehot_matrix = oneHotEncoding(column_data, classes)
        numClasses = length(classes)

        for i in 1:numClasses
            class = classes[i]
            new_col_name = String("$(col_name)_$(class)")
            dataset[!,Symbol(new_col_name)] = Int64.(onehot_matrix[:,i])
            push!(onehot_columns, new_col_name)
        end

        select!(dataset, Not(Symbol(col_name)))
    end


##############################################################################################################

    # Now, study the correlations for each feature with Churn Value

    # Calculate the correlation between each feature and the Churn Value
    correlations_with_churn = calculate_correlations_with_churn(dataset)

    # We will only use the features with a correlation value greater than a threshold (default:0.2)
    cor_with_churn_treshold = 0.2
    println("\nFeatures with a correlation value with Churn Value smaller than ", cor_with_churn_treshold, " will be removed.")
    removed_columns = find_weak_correlations_with_churn(correlations_with_churn, cor_with_churn_treshold)
    # Remove the non-correlated features:
    select!(dataset, Not(removed_columns))

    # Print the remaining features
    println("REMAINING FEATURES AND THEIR CORRELATION WITH CHURN VALUE")
    remaining = calculate_correlations_with_churn(dataset)
    rounded_df = copy(remaining)  
    for col in names(remaining)
        if eltype(remaining[!, col]) <: AbstractFloat
            rounded_df[!, col] .= round.(remaining[!, col], digits=3)
        end
    end
    println(rounded_df)

##########################################################################################

# Now, study correlations among the remaining variables

    # Get the correlation matrix
    correlation_matrix = cor(Matrix(dataset))
    variable_names = names(dataset)
    cor_among_features_treshold = 0.8

    # Get the strongly correlated features (default threshold: 0.8)
    println("\n\n--------------------------------------------------------------------------")
    println("\nNow, all the remaining features with a correlation greater than ", cor_among_features_treshold, " among them will be shown:\n")
    strong_correlations = find_strong_correlations_among_features(correlation_matrix, variable_names, cor_among_features_treshold)

     for (var1, var2, cor) in strong_correlations
        println("Correlation between $var1 y $var2 is: $(round(cor, digits=2))")
    end

    # Here, the variables to be removed must be selected by the user writing their names
    # Our selection is justified in the memory.
    removed_columns_2 = Symbol.(["Internet Service", "Dependents", "City"])

    println("\nRemoved Features: (Selected inside the code)")
    for col in removed_columns_2
        println(col)
    end
    println(length(removed_columns_2), " features were removed.\n\n")

    select!(dataset, Not(removed_columns_2))

#########################################################################################33

    # Split the dataset into inputs and targets 
    input_dataframe = select(dataset, Not(Symbol("Churn Value")))
    targets = dataset[!, Symbol("Churn Value")]

    @assert (size(input_dataframe,1)==size(targets,1)) "Matrix dimensions must match";

    return (input_dataframe, targets)

end
