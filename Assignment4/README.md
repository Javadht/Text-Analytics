# msci-text-analytics-s20
## Assignemnt4
### Result Report
In the following table you can see Accuracy Scores, from different built models:
Keep in mind that my model was better with keeping stopwords.
RelU  | tanh  |  Sigmoid
------------- | ------------- | ------------- 
Accuracy:  0.5043  | Accuracy: 0.7296 | Accuracy:  0.6886

According to the results, we can see that "tanh" has performed a better job with 73 % accuracy. 
adding L2 regularization this technique discourages learning a more complex or flexible model, avoiding the risk of Overfitting.
I used l2 regularization of 0.01 which also increased my accuracy scores in different models.
About dropout, I used 0.25 as the value of hyperparameter which also brought less loss and high accuracy, I tried more than 0.25 which gave me worst results(for instance, for dropout of 0.3 loss increased and accuracy decreased).
The reason is the following:

- In dropout method we are adding noises and making some modification over neural network itsel to prevent overfitting, if we do it more than a threshold(0.25) the generalization of the model won't work properly.
- In L2 regularization, modification occurs on the activation function itself. L2 regularization by keeping the values of the weights and biases small tries to prevent overfitting.




Note: in running inference you should give two command line arguments as below:
Terminal ----> python inference.py ./filename.txt 'relu'


