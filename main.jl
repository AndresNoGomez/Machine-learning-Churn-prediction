# ----------------- Predicting Customer Churn Using Machine Learning -----------------

# Authors:
# - Pablo Fuentes Chemes - p.fuentes@udc.es
# - Andrés Nó Gómez - andres.no.gomez@udc.es
# - Pablo Páramo Telle - pablo.paramo.telle@udc.es
# - Açelya Yildirim - acelya.yldrm@udc.es

# Firts, install the required packages if you haven't already
# using Pkg
# Pkg.add("CSV")
# Pkg.add("DataFrames")
# Pkg.add("ScikitLearn")
# Pkg.add("Flux")
# Pkg.add("CUDA")
# Pkg.add("cuDNN")
# Pkg.add("HypothesisTests")

# Load the required packages 
using CSV
using DataFrames
using Random
using Flux
using Statistics
using ScikitLearn
using CUDA
using cuDNN
using HypothesisTests
using Plots
gr()
@sk_import svm: SVC
@sk_import tree: DecisionTreeClassifier
@sk_import neighbors: KNeighborsClassifier
@sk_import decomposition:PCA
@sk_import neural_network: MLPClassifier
@sk_import ensemble:VotingClassifier
@sk_import ensemble:StackingClassifier

# Set the seed for reproducibility
Random.seed!(1); 

# Use the GPU if available to train ANNs
if CUDA.functional()
    println("CUDA is available, using the GPU to train the ANNs")
    use_gpu = true
else
    println("CUDA is not available, using the CPU to train the ANNs")
    use_gpu = false
end

# Load the functions developed on the practices and the different approaches
include("Utils/utils.jl")

include("Approaches/Approaches.jl")
include("Approaches/Approach1.jl")
include("Approaches/Approach2.jl")
include("Approaches/Approach4.jl")

include("Utils/Correlations.jl")

# Load the dataset and remove the columns that are not needed
dataset = CSV.read("Dataset/telco_churn_data.csv", DataFrame)
select!(dataset, Not([Symbol("Customer ID"), Symbol("Churn Category"), Symbol("Churn Reason"), Symbol("Customer Satisfaction")]))


# Validation ratio and folds
validationRatio = 0.2;
numfolds = 10;

# Execute the models with the different approaches
function execute_approaches(inputs, targets, global_hyperparameters; use_gpu::Bool=false, pca::Bool=false)
    results = []
    indexVector = crossvalidation(targets, numfolds)

    # ------------------------- ANN -------------------------
    println("\n\n ############### ANN ###############")
    ann_params = global_hyperparameters[:ANN]
    for i in eachindex(ann_params[:topologies_vector])
        println("**********************************\nANN model number $(i). Parameters:")
        println("Topology: $(ann_params[:topologies_vector][i])")
        println("MaxEpochs: $(ann_params[:maxEpochs_vector][i])")
        println("Learning rate: $(ann_params[:learningRate_vector][i])\n")

        modelHyperparametersANN = Dict(
            :topology => ann_params[:topologies_vector][i],
            :validationRatio => validationRatio,
            :maxEpochs => ann_params[:maxEpochs_vector][i],
            :learningRate => ann_params[:learningRate_vector][i],
            :maxEpochsVal => ann_params[:maxEpochsVal],
            :minLoss => ann_params[:minLoss],
            :numExecutions => ann_params[:numRepetitionsANNTraining]
        )

        acc_mean, acc_std, sstvt_mean, sstvt_std = modelCrossValidation(:ANN, modelHyperparametersANN, inputs, targets, indexVector; gpu=use_gpu, pca=pca)
        push!(results, (:MLP, modelHyperparametersANN, acc_mean, i))
    end

    # ------------------------- SVM -------------------------
    println("\n\n ############### SVM ###############")
    svm_params = global_hyperparameters[:SVM]
    for i in eachindex(svm_params[:kernel])
        println("**********************************\nSVM model number $(i). Parameters:")
        kern = svm_params[:kernel][i]
        println("Kernel: $kern")
        if kern == "poly"
            println("Degree of the kernel: $(svm_params[:degreeKernel][i])")
        end
        if kern in ["rbf", "poly"]
            println("Gamma of the kernel: $(svm_params[:gammaKernel][i])")
        end
        println("C value: $(svm_params[:C][i])\n")

        modelHyperparametersSVM = Dict(
            :kernel => svm_params[:kernel][i],
            :degreeKernel => svm_params[:degreeKernel][i],
            :gammaKernel => svm_params[:gammaKernel][i],
            :C => svm_params[:C][i]
        )

        acc_mean, acc_std, sstvt_mean, sstvt_std = modelCrossValidation(:SVM, modelHyperparametersSVM, inputs, targets, indexVector; pca=pca)
        push!(results, (:SVM, modelHyperparametersSVM, acc_mean, i))
    end

    # ------------------------- Decision Tree -------------------------
    println("\n\n ############### Decision Tree ###############")
    dt_params = global_hyperparameters[:Decision_tree]
    for i in eachindex(dt_params[:maxDepth])
        println("**********************************\nDecision Tree model number $(i). Parameters:")
        println("Maximum depth: $(dt_params[:maxDepth][i])")

        modelHyperparametersDT = Dict(:maxDepth => dt_params[:maxDepth][i])

        acc_mean, acc_std, sstvt_mean, sstvt_std = modelCrossValidation(:Decision_tree, modelHyperparametersDT, inputs, targets, indexVector; pca=pca)
        push!(results, (:DecisionTree, modelHyperparametersDT, acc_mean, i))
    end

    # ------------------------- KNN -------------------------
    println("\n\n ############### KNN ###############")
    knn_params = global_hyperparameters[:KNN]
    for i in eachindex(knn_params[:kValue])
        println("**********************************\nkNN model number $(i). Parameters:")
        println("Number of neighbors: $(knn_params[:kValue][i])")

        modelHyperparametersKNN = Dict(:kValue => knn_params[:kValue][i])

        acc_mean, acc_std, sstvt_mean, sstvt_std = modelCrossValidation(:KNN, modelHyperparametersKNN, inputs, targets, indexVector; pca=pca)
        push!(results, (:KNN, modelHyperparametersKNN, acc_mean, i))
    end


    # ------------------------- ENSEMBLE -------------------------
    println("\n\n ############### ENSEMBLE ###############")
    
    # Select the best models for the ensemble
    models_to_select = 4

    # Sort the results by the accuracy
    ensemble_models = sort(results, by = x -> x[3], rev = true)[1:models_to_select]
    println("Selected models for the ensemble:")
    for model in ensemble_models
        println("Model: $(model[1]) - $(model[4]), Acc: $(model[3])")
    end

    # Prepare the hyperparameters and estimators in the correct format
    hyperparameters = Vector{Dict{Any, Any}}()
    for i in eachindex(ensemble_models)
        push!(hyperparameters, ensemble_models[i][2])
    end
    estimators = [model[1] for model in ensemble_models]
    
    println("\n\n--- Soft Voting Classifier ---")
    acc_mean, acc_std, sstvt_mean, sstvt_std = trainClassEnsemble(estimators, hyperparameters, (inputs, Bool.(reshape(targets, :, 1))), indexVector, :soft; pca = pca)

    println("\n\n--- Stacking Classifier ---")
    acc_mean, acc_std, sstvt_mean, sstvt_std = trainClassEnsemble(estimators, hyperparameters, (inputs, Bool.(reshape(targets, :, 1))), indexVector, :stacking; pca = pca)
end


################ Approach 1 ################
println("----------- Approach 1 -----------")

# Process the dataset
inputs_dataframe, targets = approach1(dataset)
println(describe(inputs_dataframe))

# Execute the approach
execute_approaches(Float32.(Matrix(inputs_dataframe)), targets, obtain_hyperparameters(); use_gpu=use_gpu)



# ################ Approach 2 ################
println("----------- Approach 2 -----------")

# Process the dataset
inputs_dataframe, targets = approach2(dataset)
println(describe(inputs_dataframe))

# Execute the approach
execute_approaches(Float32.(Matrix(inputs_dataframe)), targets, obtain_hyperparameters(); use_gpu=use_gpu)



################ Approach 3 ################
println("----------- Approach 3 -----------")

# Process the dataset with the best approach, 2, and apply PCA
inputs_dataframe, targets = approach2(dataset) 

# Execute the approach with the pca transformation
execute_approaches(Float32.(Matrix(inputs_dataframe)), targets, obtain_hyperparameters(); use_gpu=use_gpu, pca=true)



################ Approach 4 ################
println("----------- Approach 4 -----------")
inputs_dataframe, targets = approach4(dataset)
println("Final inputs for approach 4:")
println(describe(inputs_dataframe))

execute_approaches(Float32.(Matrix(inputs_dataframe)), targets, obtain_hyperparameters(); use_gpu=use_gpu)


################# Final Model Training #################
#inputs_dataframe, targets = approach4(dataset) # AQUI PONER LA MEJOR APROXIMACION
#train_index, test_index = holdOut(length(targets), 0.2)

# Build the final model

# Train the final model

# Confusion matrix with the test set