using ScikitLearn
using Flux
using Flux.Losses
using Statistics
using LinearAlgebra
using Random;
using DelimitedFiles

# ---- Hold Out ---

function holdOut(N::Int, P::Float64)
    indices = randperm(N)
    num_test_patterns = round(Int, N * P)
    test_indices = indices[1:num_test_patterns]
    train_indices = indices[(num_test_patterns + 1):end]

    return (train_indices, test_indices)
end

function holdOut(N::Int, Pval::Real, Ptest::Real)
    indices = randperm(N)
    num_test_patterns = round(Int, N * Ptest)
    test_indices = indices[1:num_test_patterns]
    
    trainingValidationIndices = indices[(num_test_patterns + 1):end]
    num_val_patterns = round(Int, Pval * N / length(trainingValidationIndices))
    
    val_indices = trainingValidationIndices[1:num_val_patterns]
    train_indices = trainingValidationIndices[(num_val_patterns + 1):end]
    
    return (train_indices, val_indices, test_indices)
end


# ---- MinMaxNormalization ---

# Calculate the minimum and maximum values of each feature
function calculateMinMaxNormalizationParameters(feature::AbstractArray{<:Real,2})
    minimos = minimum(feature, dims=1)
    maximos = maximum(feature, dims=1)
    return minimos, maximos
end

# Normalize a feature matrix using Min-Max normalization and the given parameters
function normalizeMinMax(feature::AbstractArray{<:Real,2}, normalizationParameters::NTuple{2, AbstractArray{<:Real,2}})
    minimos, maximos = normalizationParameters
    feature = (feature .- minimos) ./ (maximos - minimos)
    return feature
end 

# Normalize a feature matrix using Min-Max normalization and the given parameters modifying the original matrix
function normalizeMinMax!(feature::AbstractArray{<:Real,2}, normalizationParameters::NTuple{2, AbstractArray{<:Real,2}})
    minimos, maximos = normalizationParameters
    feature .= (feature .- minimos) ./ (maximos - minimos)
    return feature
end 

# Normalize a feature matrix using Min-Max normalization and calculating the parameters
function normalizeMinMax(feature::AbstractArray{<:Real,2})
    minimos, maximos = calculateMinMaxNormalizationParameters(feature)
    feature = (feature .- minimos) ./ (maximos - minimos)
    return feature
end 

# Normalize a feature matrix using Min-Max normalization and calculating the parameters modifying the original matrix
function normalizeMinMax!(feature::AbstractArray{<:Real,2})
    normalizeMinMax!(feature, calculateMinMaxNormalizationParameters(feature))
end



# ---- ZeroMeanNormalization ---

# Calculate the mean and standard deviation of each feature
function calculateZeroMeanNormalizationParameters(feature::AbstractArray{<:Real,2})
    medias = mean(feature, dims=1)
    dt = std(feature, dims=1)
    return medias, dt
end

# Normalize a feature matrix using Zero-Mean normalization and the given parameters
function normalizeZeroMean(feature::AbstractArray{<:Real,2}, normalizationParameters::NTuple{2, AbstractArray{<:Real,2}})
    medias, dt = normalizationParameters
    feature = (feature .- medias) ./ dt
    return feature
end

# Normalize a feature matrix using Zero-Mean normalization and the given parameters modifying the original matrix
function normalizeZeroMean!(feature::AbstractArray{<:Real,2}, normalizationParameters::NTuple{2, AbstractArray{<:Real,2}})
    medias, dt = normalizationParameters
    feature .= (feature .- medias) ./ dt
    return feature
end

# Normalize a feature matrix using Zero-Mean normalization and calculating the parameters
function normalizeZeroMean(feature::AbstractArray{<:Real,2})
    medias, dt = calculateZeroMeanNormalizationParameters(feature)
    feature = (feature .- medias) ./ dt
    return feature
end

# Normalize a feature matrix using Zero-Mean normalization and calculating the parameters modifying the original matrix
function normalizeZeroMean!(feature::AbstractArray{<:Real,2})
    normalizeZeroMean!(feature, calculateZeroMeanNormalizationParameters(feature))
end




# ---- One hot encoding ----

# One hot encoding for a feature vector with classes
function oneHotEncoding(feature::AbstractArray{<:Any,1},
    classes::AbstractArray{<:Any,1})
    if length(classes)==2
        featurem = feature .== unique(classes)[1]
        featurem = reshape(featurem, (length(featurem), 1))
    elseif length(classes)>2
        n_patterns = size(feature, 1)
        n_classes = length(classes)
        featurem = zeros(Bool, n_patterns, n_classes)
        for i in 1:n_classes
            featurem[:,i] = feature .== classes[i]   
        end
    end
    return featurem
end

# One hot encoding for a feature vector
function oneHotEncoding(feature::AbstractArray{<:Any,1})
    oneHotEncoding(feature::AbstractArray{<:Any, 1}, unique(feature));
end


# One hot encoding for a feature boolean vector
function oneHotEncoding(feature::AbstractArray{Bool, 1})
        reshape(feature, (length(feature), 1));
end



# ---- Classify outputs ----

# Classify the outputs of a neural network
function classifyOutputs(outputs::AbstractArray{<:Real,2}; 
    threshold::Real=0.5) 
    if size(outputs, 2) == 1
        outputs = outputs .>= threshold
    else
        (_,indicesMaxEachInstance) = findmax(outputs, dims=2); 
        outputs = falses(size(outputs))
        outputs[indicesMaxEachInstance] .= true
    end
    return outputs
end



# ---- Accuracy ----

# Calculate the accuracy of the outputs, two classes
function accuracy(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1}) 
    return mean(targets .== outputs)
end

# Calculate the accuracy of the outputs, multiple classes
function accuracy(outputs::AbstractArray{Bool,2}, targets::AbstractArray{Bool,2}) 
    @assert(all(size(outputs).==size(targets)));
    if (size(targets,2)==1)
        return accuracy(outputs[:,1], targets[:,1]);
    else
        return mean(all(targets .== outputs, dims=2));
    end;
end

# Calculate the accuracy of the outputs, two classes with threshold
function accuracy(outputs::AbstractArray{<:Real,1}, targets::AbstractArray{Bool,1}; threshold::Real=0.5)
    return accuracy(classifyOutputs(outputs, threshold=threshold), targets)
end

# Calculate the accuracy of the outputs, multiple classes with threshold
function accuracy(outputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,2}; threshold::Real=0.5)
    @assert(all(size(outputs).==size(targets)));
    if (size(targets,2)==1)
        return accuracy(outputs[:,1], targets[:,1]);
    else
        return accuracy(classifyOutputs(outputs; threshold=threshold), targets);
    end;
end;



# ---- ANN ----

# Build a classification ANN
function buildClassANN(numInputs::Int, topology::AbstractArray{<:Int,1}, numOutputs::Int;
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology))) 
    ann=Chain();
    numInputsLayer = numInputs;
    for numHiddenLayer in 1:length(topology)
            numNeurons = topology[numHiddenLayer];
            ann = Chain(ann..., Dense(numInputsLayer, numNeurons, transferFunctions[numHiddenLayer]));
            numInputsLayer = numNeurons;
    end;
    if (numOutputs == 1)
            ann = Chain(ann..., Dense(numInputsLayer, 1, σ));
    else
            ann = Chain(ann..., Dense(numInputsLayer, numOutputs, identity));
            ann = Chain(ann..., softmax);
    end;
    return ann;
end;


# Train a classification ANN
function trainClassANN(topology::AbstractArray{<:Int,1},  
    trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}; 
    validationDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}} = 
            (Array{eltype(trainingDataset[1]),2}(undef,0,0), falses(0,0)), 
    testDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}} = 
            (Array{eltype(trainingDataset[1]),2}(undef,0,0), falses(0,0)), 
    transferFunctions::AbstractArray{<:Function,1} = fill(σ, length(topology)), 
    maxEpochs::Int = 1000, minLoss::Real = 0.0, learningRate::Real = 0.01,  
    maxEpochsVal::Int = 20, showText::Bool = false)

    # Extract the datasets
    (trainingInputs, trainingTargets) = trainingDataset
    (validationInputs, validationTargets) = validationDataset
    (testInputs, testTargets) = testDataset

    @assert size(trainingInputs,1) == size(trainingTargets,1) "Mismatch in training inputs and targets rows"

    # Initialize the ANN
    ann = buildClassANN(size(trainingInputs,2), topology, size(trainingTargets,2); transferFunctions=transferFunctions)
    loss(model, x, y) = (size(y,1) == 1) ? Losses.binarycrossentropy(model(x), y) : Losses.crossentropy(model(x), y)
    opt_state = Flux.setup(Adam(learningRate), ann)

    # Initialize early stopping variables
    bestModel = deepcopy(ann)
    bestValLoss = Inf
    epochsWithoutImprovement = 0

    # Initialize vectors to store losses
    trainingLosses = Float32[]
    validationLosses = isempty(validationInputs) ? Float32[] : Float32[NaN]
    testLosses = isempty(testInputs) ? Float32[] : Float32[NaN]


    # Initial losses
    push!(trainingLosses, loss(ann, trainingInputs', trainingTargets'))
    if !isempty(validationInputs)
        push!(validationLosses, loss(ann, validationInputs', validationTargets'))
    end
    if !isempty(testInputs)
        push!(testLosses, loss(ann, testInputs', testTargets'))
    end

    # Start the training
    for epoch in 1:maxEpochs
        # Train the ANN with the dataset and store training loss
        Flux.train!(loss, ann, [(trainingInputs', trainingTargets')], opt_state)
        trainLoss = loss(ann, trainingInputs', trainingTargets')
        push!(trainingLosses, trainLoss)

        # Early stopping based on validation loss
        if !isempty(validationInputs)
            valLoss = loss(ann, validationInputs', validationTargets')
            push!(validationLosses, valLoss)

            if valLoss < bestValLoss
                bestValLoss = valLoss
                bestModel = deepcopy(ann)
                epochsWithoutImprovement = 0
            else
                epochsWithoutImprovement += 1
            end

            # Stop if no improvement for maxEpochsVal epochs
            if epochsWithoutImprovement >= maxEpochsVal
                return bestModel, trainingLosses, validationLosses, testLosses
            end
        else
            bestModel = ann
        end

        # Calculate test loss if provided
        if !isempty(testInputs)
            push!(testLosses, loss(ann, testInputs', testTargets'))
        end

        # Print the losses if showText is true
        if showText
            println("Epoch: $epoch, Training Loss: $trainLoss" *
                    (!isempty(validationInputs) ? ", Validation Loss: $(validationLosses[end])" : "") *
                    (!isempty(testInputs) ? ", Test Loss: $(testLosses[end])" : ""))
        end

        # Stop if minimum training loss reached
        if trainLoss <= minLoss
            return bestModel, trainingLosses, validationLosses, testLosses
        end
    end

    # Return the best model and loss histories
    return bestModel, trainingLosses, validationLosses, testLosses
end



# Train a classification ANN on the GPU
function trainClassANNGPU(topology::AbstractArray{<:Int,1},  
    trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}; 
    validationDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}} = 
            (Array{eltype(trainingDataset[1]),2}(undef,0,0), falses(0,0)), 
    testDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}} = 
            (Array{eltype(trainingDataset[1]),2}(undef,0,0), falses(0,0)), 
    transferFunctions::AbstractArray{<:Function,1} = fill(σ, length(topology)), 
    maxEpochs::Int = 1000, minLoss::Real = 0.0, learningRate::Real = 0.01,  
    maxEpochsVal::Int = 20, showText::Bool = false)

    # Extract the datasets
    (trainingInputs, trainingTargets) = (gpu(trainingDataset[1]), gpu(trainingDataset[2]))
    (validationInputs, validationTargets) = (gpu(validationDataset[1]), gpu(validationDataset[2]))
    (testInputs, testTargets) = (gpu(testDataset[1]), gpu(testDataset[2]))


    @assert size(trainingInputs,1) == size(trainingTargets,1) "Mismatch in training inputs and targets rows"

    # Initialize the ANN
    ann = gpu(buildClassANN(size(trainingInputs,2), topology, size(trainingTargets,2); transferFunctions=transferFunctions))
    loss(model, x, y) = (size(y,1) == 1) ? Losses.binarycrossentropy(model(x), y) : Losses.crossentropy(model(x), y)
    opt_state = Flux.setup(Adam(learningRate), ann)

    # Initialize early stopping variables
    bestModel = deepcopy(ann)
    bestValLoss = Inf
    epochsWithoutImprovement = 0

    # Initialize vectors to store losses
    trainingLosses = Float32[]
    validationLosses = isempty(validationInputs) ? Float32[] : Float32[NaN]
    testLosses = isempty(testInputs) ? Float32[] : Float32[NaN]


    # Initial losses
    push!(trainingLosses, loss(ann, trainingInputs', trainingTargets'))
    if !isempty(validationInputs)
        push!(validationLosses, loss(ann, validationInputs', validationTargets'))
    end
    if !isempty(testInputs)
        push!(testLosses, loss(ann, testInputs', testTargets'))
    end

    # Start the training
    for epoch in 1:maxEpochs
        # Train the ANN with the dataset and store training loss
        Flux.train!(loss, ann, [(trainingInputs', trainingTargets')], opt_state)
        trainLoss = loss(ann, trainingInputs', trainingTargets')
        push!(trainingLosses, trainLoss)

        # Early stopping based on validation loss
        if !isempty(validationInputs)
            valLoss = loss(ann, validationInputs', validationTargets')
            push!(validationLosses, valLoss)

            if valLoss < bestValLoss
                bestValLoss = valLoss
                bestModel = deepcopy(ann)
                epochsWithoutImprovement = 0
            else
                epochsWithoutImprovement += 1
            end

            # Stop if no improvement for maxEpochsVal epochs
            if epochsWithoutImprovement >= maxEpochsVal
                return cpu(bestModel), trainingLosses, validationLosses, testLosses
            end
        else
            bestModel = deepcopy(ann)
        end

        # Calculate test loss if provided
        if !isempty(testInputs)
            push!(testLosses, loss(ann, testInputs', testTargets'))
        end

        # Print the losses if showText is true
        if showText
            println("Epoch: $epoch, Training Loss: $trainLoss" *
                    (!isempty(validationInputs) ? ", Validation Loss: $(validationLosses[end])" : "") *
                    (!isempty(testInputs) ? ", Test Loss: $(testLosses[end])" : ""))
        end

        # Stop if minimum training loss reached
        if trainLoss <= minLoss
            return cpu(bestModel), trainingLosses, validationLosses, testLosses
        end
    end

    # Return the best model and loss histories
    return cpu(bestModel), trainingLosses, validationLosses, testLosses
end



# ---- Confusion matrix ----

# Calculate the confusion matrix for two classes
function confusionMatrix(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1})
    # Number of patterns
    n = length(outputs)
    
    # Calculate True Positives (TP), False Positives (FP), True Negatives (TN), and False Negatives (FN)
    TP = sum(outputs .& targets)
    FP = sum(outputs .& .!targets)
    TN = sum(.!outputs .& .!targets)
    FN = sum(.!outputs .& targets)

    # Calculate accuracy and error rate
    accuracy = (TP + TN) / n
    errorRate = (FP + FN) / n
    
    # Calculate sensitivity, specificity, positive predictive value (PPV), and negative predictive value (NPV)
    sensitivity = ifelse(TP + FN == 0, 0, TP / (TP + FN))
    PPV = ifelse(TP + FP == 0, 0, TP / (TP + FP))
    specificity = ifelse(TN + FP == 0, 0, TN / (TN + FP))
    NPV = ifelse(TN + FN == 0, 0, TN / (TN + FN))

    # Calculate F1-score
    F1 = ifelse(sensitivity + PPV == 0, 0, 2 * (sensitivity * PPV) / (sensitivity + PPV))

    # Create confusion matrix
    confusionMatrix = [TN FP; FN TP]
    
    return (accuracy, errorRate, sensitivity, specificity, PPV, NPV, F1, confusionMatrix)
end

# Calculate the confusion matrix threshold
function confusionMatrix(outputs::AbstractArray{<:Real},targets::AbstractArray{Bool,1}; threshold::Real=0.5)
    binary_outputs = outputs .>= threshold  # Converts the real-valued outputs to binary
    return confusionMatrix(binary_outputs, targets)  
end


# Auxiliar to print of the confusion matrix
function printformat(accuracy, errorRate, sensitivity, specificity, PPV, NPV, F1, confusionmatrix)
    println("Accuracy: ", accuracy)
    println("Error rate: ", errorRate)
    println("Sensitivity: ", sensitivity)
    println("Specificity: ", specificity)
    println("PPV: ", PPV)
    println("NPV: ", NPV)
    println("F1: ", F1)
    println("Confusion Matrix:")
    println("| $(rpad(" ", 5)) | $(rpad("N", 5)) | $(rpad("P", 5)) |")
    println("|-------|-------|-------|")
    print("| $(rpad("N", 5)) | ")
    for element in confusionmatrix[1,:] 
        print(rpad(string(element), 5))
        print(" | ")
    end
    println()
    print("| $(rpad("P", 5)) | ")
    for element in confusionmatrix[2,:] 
        print(rpad(string(element), 5))
        print(" | ")
    end
    println()
end;

# Print the confusion matrix for two classes
function printConfusionMatrix(outputs::AbstractArray{Bool,1},targets::AbstractArray{Bool,1})
    accuracy, errorRate, sensitivity, specificity, PPV, NPV, F1, confusionmatrix = confusionMatrix(outputs, targets)
    
    # Print the metrics
    printformat(accuracy, errorRate, sensitivity, specificity, PPV, NPV, F1, confusionmatrix)
end

# Print the confusion matrix for two classes with threshold
function printConfusionMatrix(outputs::AbstractArray{<:Real},targets::AbstractArray{Bool,1}; threshold::Real=0.5)
    accuracy, errorRate, sensitivity, specificity, PPV, NPV, F1, confusionmatrix = confusionMatrix(outputs, targets; threshold=threshold)

    # Print the metrics
    printformat(accuracy, errorRate, sensitivity, specificity, PPV, NPV, F1, confusionmatrix)
end

# Calculate the confusion matrix for multiple classes 
function confusionMatrix(outputs::AbstractArray{Bool,2}, targets::AbstractArray{Bool,2}; weighted::Bool=true)
    # Check that the number of columns of both matrices is equal and different from 2
    if size(outputs, 2) != size(targets, 2) || size(outputs, 2) == 2
        error("The number of columns of both matrices must be equal and different from 2.")
    end
    
    # If both matrices have only one column, call the previous confusionMatrix function
    if size(outputs, 2) == 1
        return confusionMatrix(vec(outputs), vec(targets))
    end
    
    # Reserve memory 
    num_classes = size(outputs, 2)
    sensitivity = zeros(Float64, num_classes)
    specificity = zeros(Float64, num_classes)
    PPV = zeros(Float64, num_classes)
    NPV = zeros(Float64, num_classes)
    F1 = zeros(Float64, num_classes)
    cm = zeros(Int, num_classes, num_classes) 

    # Iterate for each class
    for i in 1:num_classes
        if sum(targets[:, i]) > 0
            metrics = confusionMatrix(outputs[:, i], targets[:, i])  
            sensitivity[i], specificity[i], PPV[i], NPV[i], F1[i] = metrics
        end
    end
        
    # Fill the confusion matrix
    for i in 1:num_classes
        for j in 1:num_classes
            cm[i, j] = sum(outputs[:, i] .& targets[:, j])
        end
    end


    # Aggregate metrics
    if weighted
        weights = vec(sum(targets, dims=1) ./ size(targets, 1))  # Proporción de muestras por clase
        sensitivity = sum(sensitivity .* weights)
        specificity = sum(specificity .* weights)
        PPV = sum(PPV .* weights)
        NPV = sum(NPV .* weights)
        F1 = sum(F1 .* weights)
    else
    sensitivity = mean(sensitivity)
    specificity = mean(specificity)
    PPV = mean(PPV)
    NPV = mean(NPV)
    F1 = mean(F1)
        sensitivity = mean(sensitivity)
        specificity = mean(specificity)
        PPV = mean(PPV)
        NPV = mean(NPV)
        F1 = mean(F1)
    end
    
    # Calculate accuracy and error rate
    ac = accuracy(outputs, targets)
    errorRate = 1 - ac

    return (ac, errorRate, sensitivity, specificity, PPV, NPV, F1, cm) 
end
# Calculate the confusion matrix for multiple classes with threshold
function confusionMatrix(outputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,2}; weighted::Bool=true)
    # Convert real-valued outputs to boolean using classifyOutputs
    boolean_outputs = classifyOutputs(outputs)

    # Call the previous confusionMatrix function with boolean matrices
    return confusionMatrix(boolean_outputs, targets; weighted=weighted)
    end

# Calculate the confusion matrix for multiple classes with threshold
function confusionMatrix(outputs::AbstractArray{<:Any,1}, targets::AbstractArray{<:Any,1}; weighted::Bool=true)
    # Ensure all output classes are included in the target classes
    @assert(all([in(output, unique(targets)) for output in outputs]))


    # Get unique classes from both outputs and targets
    classes = unique(vcat(outputs, targets))

    # Encode outputs and targets using one-hot encoding
    encoded_outputs = oneHotEncoding(outputs, classes)
    encoded_targets = oneHotEncoding(targets, classes)

    # Call the previous confusionMatrix function
    return confusionMatrix(encoded_outputs, encoded_targets; weighted=weighted)
end


# ---- Crossvalidation ----

# Crossvalidation
function crossvalidation(N::Int64, k::Int64)
    indices = repeat(1:k, Int64(ceil(N/k)));
    indices = indices[1:N];
    shuffle!(indices);
    return indices;
end;


# Crossvalidation for multiple classes and boolean targets
function crossvalidation(targets::AbstractArray{Bool,2}, k::Int64)
    # Generate the indices
    indices = Array{Int64,1}(undef, size(targets,1));

    # Generate the indices for each class
    for numClass in 1:size(targets,2)
        indices[targets[:,numClass]] = crossvalidation(sum(targets[:,numClass]), k);
    end;
    return indices;
end;


# Crossvalidation for multiple classes
function crossvalidation(targets::AbstractArray{<:Any,1}, k::Int64)
    indices = Array{Int64,1}(undef, length(targets));
    for class in unique(targets)
        indicesThisClass = (targets .== class);
        indices[indicesThisClass] = crossvalidation(sum(indicesThisClass), k);
    end;
    return indices;
end;



# ---- Model crossvalidation ----
function modelCrossValidation(modelType::Symbol,
    modelHyperparameters::Dict,
    inputs::AbstractArray{<:Real,2},
    targets::AbstractArray{<:Any,1},
    crossValidationIndices::Array{Int64,1};
    pca::Bool=false, gpu::Bool=false)
         

    # Check that the number of patterns is the same in inputs and targets
    @assert(size(inputs,1)==length(targets));

    numFolds = maximum(crossValidationIndices)

    testAccuracies = Array{Float64,1}(undef, numFolds);
    testSensitivities        = Array{Float64,1}(undef, numFolds);

    # First, we encode the targets with ANN
    if modelType==:ANN
        targets = oneHotEncoding(targets);
    end;

    # Start the crossvalidation
    for numFold in 1:numFolds

        # Split the data into training and test
        trainingInputs    = inputs[crossValidationIndices.!=numFold,:];
        testInputs        = inputs[crossValidationIndices.==numFold,:];
        trainingTargets   = targets[crossValidationIndices.!=numFold, :];
        testTargets       = targets[crossValidationIndices.==numFold, :];

        # Transform the data if PCA 
        if pca
            pca_t = PCA(0.95)
            fit!(pca_t, trainingInputs)
            trainingInputs = pca_t.transform(trainingInputs)
            testInputs = pca_t.transform(testInputs)
            # println("PCA transform: ", size(inputs,2), " -> ", size(trainingInputs,2))
            # println(describe(DataFrame(trainingInputs, :auto)))
        end
        
        
        # If we are using SVM, DecisionTree or kNN, we use the ScikitLearn package
        if (modelType==:SVM) || (modelType==:Decision_tree) || (modelType==:KNN)

            # We define the constructors for the models
            modelConstructors = Dict(
                :SVM => () -> SVC(kernel=modelHyperparameters[:kernel], degree=modelHyperparameters[:degreeKernel], gamma=modelHyperparameters[:gammaKernel], C=modelHyperparameters[:C]),
                :Decision_tree => () -> DecisionTreeClassifier(max_depth=modelHyperparameters[:maxDepth], random_state=1),
                :KNN => () -> KNeighborsClassifier(modelHyperparameters[:kValue])
            );

            # Train the model
            model = modelConstructors[modelType]()
            model = fit!(model, trainingInputs, vec(trainingTargets));

            # Predict the test data
            testOutputs = predict(model, testInputs);

            # Calculate the confusion matrix
            (acc, _, sens, _, _, _, _) = confusionMatrix(testOutputs, vec(testTargets));

        else

            # Ensure that the model is ANN
            @assert(modelType==:ANN);

            # Since we are using ANN, we need to train some times the model to get the average of the metrics
            testAccuraciesEachRepetition = Array{Float64,1}(undef, modelHyperparameters[:numExecutions]);
            testSensitivitiesEachRepetition         = Array{Float64,1}(undef, modelHyperparameters[:numExecutions]);

            # Train the model numExecutions times
            for numTraining in 1:modelHyperparameters[:numExecutions]

                if modelHyperparameters[:validationRatio]>0
                    # In the case of training an ANN with a validation set, we make an additional division train+val with hold out:
                    (trainingIndices, validationIndices) = holdOut(size(trainingInputs,1), modelHyperparameters[:validationRatio]*size(trainingInputs,1)/size(inputs,1));

                    # Train the ANN with the training and validation sets
                    if gpu
                        ann, trainingLosses, validationLosses, testLosses = trainClassANNGPU(modelHyperparameters[:topology], (trainingInputs[trainingIndices,:],   trainingTargets[trainingIndices,:]),
                            validationDataset = (trainingInputs[validationIndices,:], trainingTargets[validationIndices,:]),
                            testDataset = (testInputs, testTargets),
                            maxEpochs=modelHyperparameters[:maxEpochs], learningRate=modelHyperparameters[:learningRate], maxEpochsVal=modelHyperparameters[:maxEpochsVal]);
                    else
                        ann, trainingLosses, validationLosses, testLosses = trainClassANN(modelHyperparameters[:topology], (trainingInputs[trainingIndices,:],   trainingTargets[trainingIndices,:]),
                            validationDataset = (trainingInputs[validationIndices,:], trainingTargets[validationIndices,:]),
                            testDataset = (testInputs, testTargets),
                            maxEpochs=modelHyperparameters[:maxEpochs], learningRate=modelHyperparameters[:learningRate], maxEpochsVal=modelHyperparameters[:maxEpochsVal]);
                    end;

                    #= # Show the evolution of the loss
                    if (numFold==1 && numTraining == 1)                      
                        g = plot(title = "Loss evolution 1 fold", xaxis = "Epoch", yaxis = "Loss")
                        plot!(g,1:length(trainingLosses),trainingLosses,label="Train error",color="green")
                        plot!(g,1:length(testLosses),testLosses,label="Test error",color="red")
                        plot!(g,1:length(validationLosses),validationLosses,label="Val error",color="blue")   
                        display(g) 
                    end=#

                else

                    # Otherwise, train the ANN with the training set
                    if gpu
                        ann, = trainClassANNGPU(modelHyperparameters[:topology], (trainingInputs, trainingTargets),
                        testDataset = (testInputs, testTargets),
                        maxEpochs=modelHyperparameters[:maxEpochs], learningRate=modelHyperparameters[:learningRate]);
                    else
                        ann, = trainClassANN(modelHyperparameters[:topology], (trainingInputs, trainingTargets),
                        testDataset = (testInputs, testTargets),
                        maxEpochs=modelHyperparameters[:maxEpochs], learningRate=modelHyperparameters[:learningRate]);
                    end;

                end;

                # Calculate the confusion matrix
                (testAccuraciesEachRepetition[numTraining], _, testSensitivitiesEachRepetition[numTraining], _, _, _, _) = confusionMatrix(collect(ann(testInputs')'), testTargets);

            end;

            # Calculate the average of the metrics
            acc = mean(testAccuraciesEachRepetition);
            sens  = mean(testSensitivitiesEachRepetition);

        end;

        # Save the results
        testAccuracies[numFold] = acc;
        testSensitivities[numFold] = sens;

        # println("Results in test in fold ", numFold, "/", numFolds, ": accuracy: ", round(100*testAccuracies[numFold],digits=3), " %, Sensitivity: ", round(100*testSensitivities[numFold],digits=3), " %");

    end; 

    println(modelType, ": Average test accuracy on a ", numFolds, "-fold crossvalidation: ", round(mean(testAccuracies), digits=7 ), ", with a standard deviation of ", round(std(testAccuracies), digits=7));
    println(modelType, ": Average test sensitivity on a ", numFolds, "-fold crossvalidation: ", round(mean(testSensitivities), digits=7), ", with a standard deviation of ", round(std(testSensitivities),digits=7));

    return (mean(testAccuracies), std(testAccuracies), mean(testSensitivities), std(testSensitivities));

end;

# ----- Ensemble Model ------   

function trainClassEnsemble(estimators::AbstractArray{Symbol,1},
    modelsHyperParameters::AbstractArray{Dict{Any, Any}, 1},
    trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}},    
    kFoldIndices::Array{Int64,1},
    typeEnsemble::Symbol;
    pca::Bool=false)

    (inputs, targets) = trainingDataset;

    numFolds = maximum(kFoldIndices);

    # Initialize the vectors to store the results for example acc and sens
    testAccuracies = Array{Float64,1}(undef, numFolds);
    testSensitivity = Array{Float64,1}(undef, numFolds);

    for numFold in 1:numFolds
        # Split the dataset into training and test
        trainingInputs    = inputs[kFoldIndices.!=numFold,:];
        testInputs        = inputs[kFoldIndices.==numFold,:];
        trainingTargets   = vec(targets[kFoldIndices.!=numFold,:]);
        testTargets       = vec(targets[kFoldIndices.==numFold,:]);

        
        # Transform the data if PCA 
        if pca
            pca_t = PCA(0.95)
            fit!(pca_t, trainingInputs)
            trainingInputs = pca_t.transform(trainingInputs)
            testInputs = pca_t.transform(testInputs)
        end

        trained_models = Dict{String, Any}();

        # Train individual models
        for (i, (estimator, hyperParams)) in enumerate(zip(estimators, modelsHyperParameters))
            model_name = "$(Symbol(estimator))_$i"
            model = trainModel(estimator, hyperParams, trainingInputs, trainingTargets)
            acc = score(model, testInputs, testTargets)
            trained_models[model_name] = model
            #println("$model_name: $(round(acc * 100, digits=2)) %")
        end
        
        if typeEnsemble == :soft
            # Build an ensemble using a VotingClassifier (Soft Voting)
            ensemble_model = VotingClassifier(
                estimators=[(name, trained_models[name]) for name in keys(trained_models)],
                voting="soft", weights=collect(range(2, stop=1, length=length(trained_models))) # More weight to the first model (the best one)
            )
        else 
            if typeEnsemble == :stacking
                # Build an ensemble using a VotingClassifier (Stacking)
                ensemble_model = StackingClassifier(
                    estimators=[(name, trained_models[name]) for name in keys(trained_models)],
                    final_estimator=SVC(probability=true)
                )
            else
                error("Unsupported ensemble type: $typeEnsemble")
            end
        end

        fit!(ensemble_model, trainingInputs, trainingTargets)

        # we obtain the predicted outputs for the test data
        testOutputs = predict(ensemble_model, testInputs);

        # We calculate the metrics with the function developed in the previous practice
        (accuracy, _, sensitivity, _, _, _, _, _) = confusionMatrix(testOutputs, vec(testTargets));

        # println("Fold $(numFold): Ensembled model accuracy: $(round(accuracy * 100, digits=4)) %, sensitivity: $(round(sensitivity * 100, digits=4)) %");

        testAccuracies[numFold] = accuracy;
        testSensitivity[numFold] = sensitivity;

    end;

    println("Ensemble model: Average test accuracy on a ", numFolds, "-fold crossvalidation: ", round(mean(testAccuracies), digits=7 ), ", with a standard deviation of ", round(std(testAccuracies), digits=7));
    println("Ensemble model: Average test sensitivity on a ", numFolds, "-fold crossvalidation: ", round(mean(testSensitivity), digits=7), ", with a standard deviation of ", round(std(testSensitivity),digits=7));
    return mean(testAccuracies), std(testAccuracies), mean(testSensitivity), std(testSensitivity);

end
function trainModel(estimator::Symbol, hyperParams::Dict{Any, Any}, inputs, targets)
    model = nothing
    if estimator == :SVM
        model = SVC(
            kernel=hyperParams[:kernel], 
            degree=hyperParams[:degreeKernel], 
            gamma=hyperParams[:gammaKernel], 
            C=hyperParams[:C],
            probability=true
        )
    elseif estimator == :DecisionTree
        model = DecisionTreeClassifier(
            max_depth=hyperParams[:maxDepth], 
            random_state=1
        )
    elseif estimator == :KNN
        model = KNeighborsClassifier(
            hyperParams[:kValue]
        )
    elseif estimator == :MLP
        model = MLPClassifier( 
            hidden_layer_sizes=hyperParams[:topology],
            activation="relu",
            solver="adam",  
            learning_rate="constant",
            learning_rate_init = hyperParams[:learningRate],
            max_iter=hyperParams[:maxEpochs],
            early_stopping=true,
            validation_fraction=hyperParams[:validationRatio],
            n_iter_no_change=n_iter_no_change=hyperParams[:maxEpochsVal]  
        )
    else
        error("Unsupported estimator: $estimator")
    end
    fit!(model, inputs, targets)
    return model
end

