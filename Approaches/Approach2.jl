# ---------------------------------------------
# ----------------- Aproach 2 -----------------
# ---------------------------------------------

# The multi-class categorical variables will be assigned continuous values derived from metrics
# such as the proportion of each category in relation to the final outcome. The models will be evaluated using these new
# metrics to assess their effectiveness in improving performance and capturing patterns in the data.


function approach2(dataset_df::DataFrame)
    # Check if the DataFrame has the expected number of columns
    dataset = copy(dataset_df)

    expected_columns = 42  
    if size(dataset, 2) != expected_columns
        throw(ArgumentError("The DataFrame must have $expected_columns columns, but it has $(size(dataset, 2))."))
    end

    # Process binary and continuous columns
    process_binary_continuous_columns!(dataset)

    # Multicategorical variables with the percentage of churn in each category
    categorical_columns_to_transform = [
        "Offer", "Internet Type", "Contract", "Payment Method", "City", "Zip Code"
    ]
    propotion_encoder!(categorical_columns_to_transform, dataset)


    # Process the cyclical features: latitude and longitude.
    process_latitude_and_longitude!(dataset)


    # Split the dataset into inputs and targets 
    input_dataframe = select(dataset, Not(Symbol("Churn Value")))
    targets = dataset[!, Symbol("Churn Value")]

    @assert (size(input_dataframe,1)==size(targets,1)) "Matrix dimensions must match";

    return (input_dataframe, targets)
end

