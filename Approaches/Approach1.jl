# ---------------------------------------------
# ----------------- Aproach 1 -----------------
# ---------------------------------------------

# This approach is based on the idea of normalizing continuous and binary variables to a range of [0, 1] 
# and mapping categorical variables with limited categories to specific predefined values. 
# Variables with multiple categories will be preprocessed in the second approach.

function approach1(dataset_df::DataFrame)
    # Check if the DataFrame has the expected number of columns
    dataset = copy(dataset_df)

    expected_columns = 42  
    if size(dataset, 2) != expected_columns
        throw(ArgumentError("The DataFrame must have $expected_columns columns, but it has $(size(dataset, 2))."))
    end

    # Multicategorical variables with limited categories
    categorical_columns_to_transform = ["Offer", "Internet Type", "Contract", "Payment Method"]
    label_encoder!(categorical_columns_to_transform, dataset)


    # Process binary and continuous columns
    process_binary_continuous_columns!(dataset)


    # Split the dataset into inputs and targets 
    input_dataframe = select(dataset, Not([Symbol("Longitude"), Symbol("Latitude"), Symbol("Churn Value"), Symbol("City"), Symbol("Zip Code")]))
    targets = dataset[!, Symbol("Churn Value")]

    @assert (size(input_dataframe,1)==size(targets,1)) "Matrix dimensions must match";


    return (input_dataframe, targets)
end

