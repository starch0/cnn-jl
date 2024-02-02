using Pkg
Pkg.add("Images")
Pkg.add("Flux")

filesPath = ""

using Random
using Images

function loadImages(filesPaths)
    images = [Images.load(filePath) for filePath in filesPaths]
    return images 
end

#model archtecture
model = Chain(
    Conv((3,3), 1=>6, relu)
    MaxPool((2,2))
    Conv((3,3), 16=>32, relu)
    MaxPool((2,2))
    flatten,
    Dense(288,2),
    softmax
)

function splitData(images, labels, trainRatio)
    n = length(images)
    indices = randperm(n)
    trainSize = Int(round(trainRatio * n))
    trainIndices = indices[1:trainSize]
    testIndices = indices[trainSize+1:end]
    return images[trainIndices], labels[trainIndices], images[testIndices], labels[testIndices]
end

using Flux: crossentropy, ADAM

function trainModel(model, trainData, trainLabels, epochs)
    loss(x, y) = crossentropy(model(x), y)
    opt = ADAM()

    for epoch in 1:epochs
        Flux.train!(loss, Flux.params(model), zip(trainData, trainLabels), opt)
    end
end
