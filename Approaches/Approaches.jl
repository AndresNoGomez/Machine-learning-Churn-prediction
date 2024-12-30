# This file contains the functions for the processes that are common to all the approaches.

# Normalize the binary and the continuous columns. As these features are the same for all approaches,
# the name of these columns is defined inside the function.
function process_binary_continuous_columns!(dataset::DataFrame)
    # Convert the binary columns to a binary representation
    bool_columns_to_transform = [
        "Referred a Friend", "Phone Service", "Multiple Lines", "Internet Service", "Online Security", 
        "Online Backup", "Device Protection Plan", "Premium Tech Support", 
        "Streaming TV", "Streaming Movies", "Streaming Music", "Unlimited Data", 
        "Paperless Billing", "Under 30", "Senior Citizen", "Married", "Dependents"
    ]

    for col_name in bool_columns_to_transform
        dataset[!, Symbol(col_name)] .= Int.(dataset[!, Symbol(col_name)] .== "Yes")
    end

    # The gender too (it is a binary variable, Male or Female)
    dataset[!, Symbol("Gender")] .= Int.(dataset[!, Symbol("Gender")] .== "Male")



    # Normalize the numerical columns with minmax
    numerical_columns_to_normalize_minmax = ["Age", "Population"]

    for col_name in numerical_columns_to_normalize_minmax
        column_matrix = reshape(Float64.(dataset[!, Symbol(col_name)]), :, 1)
        normalizeMinMax!(column_matrix)
        dataset[!, Symbol(col_name)] .= column_matrix[:, 1]  
    end

    # Normalize the numerical clumns with meanstd
    numerical_columns_to_normalize_meanstd = [
        "Number of Referrals", "Tenure in Months", "Avg Monthly Long Distance Charges",
        "Avg Monthly GB Download", "Monthly Charge", "Total Regular Charges", "Total Refunds",
        "Total Extra Data Charges", "Total Long Distance Charges", "Number of Dependents", "CLTV",
        "Total Customer Svc Requests", "Product/Service Issues Reported"
    ]

    for col_name in numerical_columns_to_normalize_meanstd
        column_matrix = reshape(Float64.(dataset[!, Symbol(col_name)]), :, 1)
        normalizeZeroMean!(column_matrix)
        dataset[!, Symbol(col_name)] .= column_matrix[:, 1]  
    end

end

###########################################################################################################

# Encode the categorical columns with a label encoding 
function label_encoder!(categorical_columns_to_transform::Vector{String}, dataset::DataFrame)
    for col_name in categorical_columns_to_transform
        column_data = dataset[!, Symbol(col_name)]
        unique_classes = unique(column_data)
        mapping = label_encode(unique_classes) 
        dataset[!, Symbol(col_name)] .= map(x -> mapping[x], column_data)  
    end
end


# Label encoding between 0 and 1 for a set of unique classes
function label_encode(unique_classes::AbstractArray{<:Any,1})
    classes = Dict();
    n_classes = length(unique_classes);
    for (i, class) in enumerate(unique_classes)
        classes[class] = (i - 1) / (n_classes - 1);
    end
    return classes;
end;


############################################################################################################

function propotion_encoder!(categorical_columns_to_transform::Vector{String}, dataset::DataFrame)

    for col_name in categorical_columns_to_transform
        churn_count = Dict{String, Int}()
        total_count = Dict{String, Int}()

        # Count churn for each category in the column
        for i in eachindex(dataset[:, 1])
            category = string(dataset[i, Symbol(col_name)])
            churn_value = dataset[i, Symbol("Churn Value")]

            # Update the total count for the category
            total_count[category] = get(total_count, category, 0) + 1
            if churn_value == 1
                churn_count[category] = get(churn_count, category, 0) + 1
            end
        end

        # Calculate the churn percentage for each category
        churn_percentage = Dict(
            category => get(churn_count, category, 0) / total_count[category] 
            for category in keys(total_count)
        )

        # Replace the category with the churn percentage
        dataset[!, Symbol(col_name)] .= map(x -> churn_percentage[string(x)], dataset[!, Symbol(col_name)])

        # Min - Max to be in range [0, 1] and preserve the relation between the categories
        column_matrix = reshape(Float64.(dataset[!, Symbol(col_name)]), :, 1)
        normalizeMinMax!(column_matrix)
        dataset[!, Symbol(col_name)] .= column_matrix[:, 1]  

    end
end

#################################################################################################################

function process_latitude_and_longitude!(dataset::DataFrame)

    # Create the Latitude sinus and cosinus representations
    dataset[!, Symbol("Latitude_Sin")] = sin.(deg2rad.(dataset[!, Symbol("Latitude")]))
    # Create the Longitude sinus and cosinus representations
    dataset[!, Symbol("Longitude_Sin")] = sin.(deg2rad.(dataset[!, Symbol("Longitude")]))
    dataset[!, Symbol("Longitude_Cos")] = cos.(deg2rad.(dataset[!, Symbol("Longitude")]))

    # Remove the original columns
    select!(dataset, Not([:Latitude, :Longitude]))

end


# Function to obtain the hyperparameters for the models 
function obtain_hyperparameters()
    return Dict(
        :ANN => Dict(
            :topologies_vector => [[5], [10], [40], [5, 5], [20, 20], [50, 50], [50, 40], [100, 100]],
            :maxEpochs_vector => [250, 200, 300, 300, 200, 200, 200, 200],
            :learningRate_vector => [0.02, 0.02, 0.01, 0.02, 0.01, 0.01, 0.01, 0.01],
            :maxEpochsVal => 15,
            :numRepetitionsANNTraining => 10,
            :minLoss => 0,
        ),
        :SVM => Dict(   # Parameters set to 0 are not used for that type of kernel
            :kernel => ["linear", "rbf", "rbf", "rbf", "sigmoid", "poly", "poly", "poly"],
            :degreeKernel => [0, 0, 0, 0, 0, 1, 2, 3],
            :gammaKernel => [0, "auto", "scale", 1, 0, "auto", "scale", 2],
            :C => [0.1, 0.1, 1, 1, 1, 10, 1, 0.1]
        ),
        :Decision_tree => Dict(
            :maxDepth => [2, 4, 8, 16, 32, 64]
        ),
        :KNN => Dict(
            :kValue => [2, 3, 5, 10, 15, 30]
        )
    )
end
