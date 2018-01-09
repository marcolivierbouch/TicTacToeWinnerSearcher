from numpy import exp, array, random, dot


class NeuronLayer():
    def __init__(self, numberOfNeurons, numberOfInputsPerNeuron):
        self.synapticWeights = 2 * random.random((numberOfInputsPerNeuron, numberOfNeurons)) - 1


class NeuralNetwork():
    def __init__(self, layer1, layer2, layer3):
        self.layer1 = layer1
        self.layer2 = layer2
        self.layer3 = layer3

    def __sigmoid(self, x):
        return 1 / (1 + exp(-x))

    def __sigmoidDerivative(self, x):
        return x * (1 - x)

    def train(self, trainingSetInputs, trainingSetOutputs, numberOfTrainingIterations):

        for iteration in range(numberOfTrainingIterations):

            outputFromLayer1, outputFromLayer2, outputFromLayer3 = self.think(trainingSetInputs)

            layer3Error = trainingSetOutputs - outputFromLayer3
            layer3Delta = self.__calculateDelta(layer3Error, outputFromLayer3)

            layer2Error = layer3Delta.dot(self.layer3.synapticWeights.T)
            layer2Delta = self.__calculateDelta(layer2Error, outputFromLayer2)

            layer1Error = layer2Delta.dot(self.layer2.synapticWeights.T)
            layer1Delta = self.__calculateDelta(layer1Error, outputFromLayer1)

            layer1Adjustment = self.__calculateAdjustement(trainingSetInputs, layer1Delta)
            layer2Adjustment = self.__calculateAdjustement(outputFromLayer1, layer2Delta)
            layer3Adjustment = self.__calculateAdjustement(outputFromLayer2, layer3Delta)

            self.layer1.synapticWeights += layer1Adjustment
            self.layer2.synapticWeights += layer2Adjustment
            self.layer3.synapticWeights += layer3Adjustment

    def __calculateAdjustement(self, lastOutputLayer, layerDelta):
        return lastOutputLayer.T.dot(layerDelta)

    def __calculateDelta(self, layerError, outputFromLayer):
        return layerError * self.__sigmoidDerivative(outputFromLayer)

    def __calculateOutputFromLayer(self, lastLayerOutPut, layer):
        return self.__sigmoid(dot(lastLayerOutPut, layer.synapticWeights))

    def think(self, inputs):
        outputFromLayer1 = self.__calculateOutputFromLayer(inputs, self.layer1)
        outputFromLayer2 = self.__calculateOutputFromLayer(outputFromLayer1, self.layer2)
        outputFromLayer3 = self.__calculateOutputFromLayer(outputFromLayer2, self.layer3)
        return outputFromLayer1, outputFromLayer2, outputFromLayer3

    def print_weights(self):
        print ("Layer 1 (9 neurons, each with 9 inputs): ")
        print (self.layer1.synapticWeights)
        print ("Layer 2 (9 neurons, with 9 inputs):")
        print (self.layer2.synapticWeights)
        print ("layer 3 (1 neuron, with 4 inputs: )")
        print (self.layer3.synapticWeights)

if __name__ == "__main__":

    random.seed(1)

    layer1 = NeuronLayer(9, 9)
    layer2 = NeuronLayer(9, 9)
    layer3 = NeuronLayer(1, 9)
 
    neuralNetwork = NeuralNetwork(layer1, layer2, layer3)

    print ("Random starting synaptic weights: ")
    neuralNetwork.print_weights()

    trainingSetInputs = array([[0, 0, 1, 0, 0, 1, 0, 0, 0], [0, 1, 1, 0, 1, 0, 1, 0, 0], [1, 0, 1, 1, 0, 0, 1, 0, 0], [0, 1, 0, 0, 0, 0, 1, 1, 1], [1, 0, 0, 0, 1, 0, 1, 0, 1], [1, 1, 0, 0, 1, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0]])
    trainingSetOutputs = array([[0, 1, 1, 1, 1, 0, 0]]).T
    
    countOfIteration = 100000
    neuralNetwork.train(trainingSetInputs, trainingSetOutputs, countOfIteration)

    print ("New synaptic weights after training: ")
    neuralNetwork.print_weights()

    print ("New situation [1, 1, 0, 1, 1, 0, 1, 0, 0]-> ?: ")
    hiddeState, hiddeState1, output = neuralNetwork.think(array([1, 1, 0, 1, 1, 0, 1, 0, 0]))
    print (output)

    print ("New situation [1, 1, 0, 1, 1, 0, 0, 0, 0] -> ?: ")
    hiddenState, hiddeState1, output = neuralNetwork.think(array([1, 1, 0, 1, 1, 0, 0, 0, 0]))
    print (output)

    print ("New situation [1, 1, 0, 0, 0, 0, 0, 1, 1] -> ?: ")
    hiddenState, hiddeState1, output = neuralNetwork.think(array([1, 1, 0, 0, 0, 0, 0, 1, 1]))
    print (output)
    
    print ("New situation [1, 1, 0, 0, 0, 0, 1, 1, 1] -> ?: ")
    hiddenState, hiddeState1, output = neuralNetwork.think(array([1, 1, 0, 0, 0, 0, 1, 1, 1]))
    print (output)
