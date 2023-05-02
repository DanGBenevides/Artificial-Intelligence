class Perceptron:
    def __init__(self, num_inputs):
        self.weights = [0] * num_inputs
        self.bias = 0

    def predict(self, inputs):
        activation = self.bias
        for i, w in zip(inputs, self.weights):
            activation += i * w
        return 1 if activation >= 0 else 0

    def train(self, training_inputs, labels, learning_rate=0.1, max_iterations=100):
        for _ in range(max_iterations):
            for inputs, label in zip(training_inputs, labels):
                prediction = self.predict(inputs)
                self.bias += learning_rate * (label - prediction)
                for i in range(len(self.weights)):
                    self.weights[i] += learning_rate * (label - prediction) * inputs[i]

    @staticmethod
    def generate_and_data(n):
        inputs = []
        labels = []
        for i in range(2**n):
            input_bits = [int(x) for x in list(bin(i)[2:].zfill(n))]
            inputs.append(input_bits)
            labels.append(1 if all(input_bits) else 0)
        return inputs, labels

    @staticmethod
    def generate_or_data(n):
        inputs = []
        labels = []
        for i in range(2**n):
            input_bits = [int(x) for x in list(bin(i)[2:].zfill(n))]
            inputs.append(input_bits)
            labels.append(1 if any(input_bits) else 0)
        return inputs, labels

    @staticmethod
    def generate_xor_data(n):
        inputs = []
        labels = []
        for i in range(2**n):
            input_bits = [int(x) for x in list(bin(i)[2:].zfill(n))]
            inputs.append(input_bits)
            labels.append(1 if sum(input_bits) % 2 == 1 else 0)
        return inputs, labels



print("digite o numero de entradas: ")
n = int(input())
print("digite a função desejada: ")
print("1 - AND")
print("2 - OR")
print("3 - XOR")
op = int(input())
if op == 1:
    inputs, labels = Perceptron.generate_and_data(n)
elif op == 2:
    inputs, labels = Perceptron.generate_or_data(n)
elif op == 3:
    inputs, labels = Perceptron.generate_xor_data(n)
else:
    print("opção invalida")
    exit()

print("inputs: ", inputs)
print("labels: ", labels)

perceptron = Perceptron(n)
perceptron.train(inputs, labels)

# remova o comentário caso queira ver os pesos e bias
# print("pesos: ", perceptron.weights)
# print("bias: ", perceptron.bias)

print("digite as entradas: ")
inputs = []
for i in range(n):
    inputs.append(int(input()))
print("resultado: ", perceptron.predict(inputs))
