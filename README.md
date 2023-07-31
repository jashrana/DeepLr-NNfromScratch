# DeepLr NNfromScratch
 Creating a Single layer Neural Network & a Shallow Neural Network from Scratch and testing algorithms based on Stochastic Gradient Descent.

 Code Files:
* Deep Learning Assignment -1.ipynb

# University of Galway
## CT5133: Deep Learning Assignment 1 2023

Student Name(s): <b>Soumitra Koustubh Manavi, Jash Prakash Rana</b> <br>

Student ID(s): <b>22220805, 22222806</b>

## 1. Creating a Logistic Neural Network (Single Layer Perceptron) using the Stochastic Gradient Descent optimisation technique

A `single layer neural network` is a network where there is an input layer and only one node of the output layer designed into the system. No hidden layers are present, and there is only one layer of weights equivalent to the number of inputs and one bias is present to make predictions. The weights are adjusted according to some optimisation technique(in this case, Stochastic Gradient Descent) which tries to find a minimum number(in this case, the cost of the error) in order to minimise the errors and find the optimal value for the weights and the bias, which increases the chance to predict the right output in each case.

A neural network is said to be `logistic` when the sigmoid activation function(also called a special form of the logistic function) is used to determine the output label using the data fed into the network.[1] The sigmoid function can be written as:
$$ \sigma(X) = \frac{1}{1 + e^{z}} $$
where:
* $z$ = $w$.$x$ + $b$
* $w$ = number of weights (size  = number of attributes)
* $x$ = attributes
* $b$ = bias </br>

Gradient Descent is a popular optimisation technique among deep learning enthusiasts that tries to minimize the error by taking a step "downhill" in the right direction and then updating the weights and the bias according to the errors calculated. There reaches a point where the error starts to go up, which is not needed by the program, so early stopping is required to stop the program once it reaches the optimum value. One of the variations of this algorithm is the `Stochastic Gradient Descent`, where one random data row is selected to calculate the errors and perform the gradient descent algorithm. 

The error(or the loss) is calculated using the log loss function for each data point as follows[2]:
$$ Loss = 𝑦(𝑖)*log (\hat{𝑦}(𝑖)) + (1−𝑦(𝑖))*log(1 − \hat{𝑦}(𝑖)) $$
where:
* $i$ = data at the $i^th$ row
* y = actual output
* $\hat{𝑦}$ = predicted output </br>

Each data row has a different loss which is summed up and then averaged over the length of the data(number of rows present). This computation is called the `cost function` and this is the main determinant of the optimisation function. Each time the cost function is calculated for the data, it is compared to the previous cost, to stop when the cost value has reached the optimal value(minimum). The cost function is as follows[2]:
$$ Cost = J(w,b) = -\frac{1}{N} \sum^{N}_{i=1} (𝑦(𝑖)*log (\hat{𝑦}(𝑖)) + (1−𝑦(𝑖))*log(1 − \hat{𝑦}(𝑖)))$$
where:
* N = number of rows present

The weights are the number associated with each attribute which determines the best path toward the correct prediction. The bias is a constant term determined to set the slope of the linear line on the hyperplane of the model. To update them, partial derivatives are taken. Partial derivates for weight include the error in the data of each attribute, while for bias only the error is determined.
To update the weights and bias for each data set/point, the following formula is used[2]:
$$ w_(j) = w_(j) - \alpha \times \delta w $$
$$ b_(j) = b_(j) - \alpha \times \delta b $$ 
where:
* $\alpha$ = learning rate
* $\delta w$ = partial derivates for weights
* $\delta b$ = partial derivates for bias </br>


Each neural network has some hyperparameters responsible for fine-tuning the neural network model made for each scenario. They are responsible to find out the best settings in which the architecture is able to predict with the highest accuracy. The hyperparameters used are:
1. Iterations: The maximum number of times the loop will run to adjust the weights and bias until the program finds the optimum value.
2. Learning Rate: The rate at which the weights and the bias will be updated. A smaller number means smaller steps toward the optimum value and vice versa.
3. Threshold: There is a value used to activate a sigmoid neuron, called the threshold. We have considered 0.5 to be the threshold for this assignment.

## 2. Implementing a Shallow Neural Network

A `shallow neural network` is a neural network architecture with <b>1 hidden layer</b> between the inputs and the output layer. A hidden layer can have an arbitrary number of nodes connected to the inputs and can perform their own computation as per the data and weights. A <i>node</i> is a processing unit in the neural network architecture capable of computing some kind of activations with the input data and it sends it forward into the next layer(in this case, the output layer). The usual approach is to make this a feed-forward network, where the output of each layer is the input to the next layer.

The biggest advantage of increasing the size of the network is to make our model predict on non-linearly separable dataset and get a good prediction on it. The downside is that with each layer increased, there is a chance of overfitting, the computational resources increase and the complexity of the model increases meaning it becomes problematic to understand the calculations at each node in each layer.

There are 2 types of algorithms used to train the network:
1. Forward propagation: Forward propagation algorithm is responsible for putting the input through all the layers of the neural network. In practical execution, the input goes to the first layer, calculates the output at each node, and the outputs are then forwarded to the next layer, in a fully-connected fashion. The weights and biases for each node/input are different, and that is the best way to compute the predictions.
2. Backpropagation: Backpropagation is the technique in which after running the forward propagation on the data, whatever loss is calculated should have an update on the weights accordingly. Now to go back to each layer, the derivative of the sigmoid function is calculated which is used to calculate the error in each layer, and accordingly update each weight associated with each node.

The cost function is still calculated at the output as we are interested in getting the output correct i.e. minimise the errors at the output.

The input size is passed here to tell the attributes of the dataset, while the hidden size & output size is the number of neurons in the hidden layer and output layer respectively. Accordingly, the weights and biases matrix will be created for each layer.

## 3. Enhancement (Submission)
Back Prop with Momentum: 
The enhancement of using Back Prop using Momentum helps in reducing the change of getting stuck in local optima and enhances the loss to reduce up to **global optima**. There should not be high momentum as well which can add the risk of skipping the global optima during gradient descent.

On first Iteration:
$$ V_\delta w = \delta w $$
$$ V_\delta b = \delta b $$
After First Iteration:
$$ V_\delta w = (1-\beta)*\delta w + \beta * V_\delta w $$
$$ V_\delta b = (1-\beta)*\delta b + \beta * V_\delta b $$
where:
* $V$ = velocity
* $\delta w$ = Partial Derivative of Weights
* $\delta b$ = Partial Derivative of Biases
* $\beta$ = Momentum

---
References: </br>
[1] https://machinelearningmastery.com/a-gentle-introduction-to-sigmoid-function/ </br>
[2] Deep Learning, "Topic 02: Fundamentals of Neural Networks, Part 1" by Prof. Michael Madden, Chair of Computer Science, Head of Machine Learning Group, University of Galway
