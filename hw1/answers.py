r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers

part1_q1 = r"""
**False**
In-Sample error definition: The error rate we get on the same data set we used to build our predictor.
Therefore, in order to get our in-sample error, we need to run the predictor on the same data we built it on and calculate the in-sample error from the results.


**2.False**
Our wanted scenario is that the training data and testing data should be drawn from the same 
distribution. I will give a concrete example:
Lets say we are trying to build a model to classify images of cats and dogs, and lets pretend that the world's cat-dog ratio is 4:1.
Lets pretend that our database has the same distribution of labeles in the database.
If we picked a training group that consists only of cats, and a 20% size test set that consists of dogs (is a possible scenario when randomly selecting images)
our model will learn that in practice, it encounters only 1 class  - and makes the generalization assumption that in reality he should predict that the input sample is a dog all of the time.
Of course, this will mean our accuracy will be 0 or close to it when comparing against the test set, which is also not true, considering the fact that in the real world, we would have an 80% chance to guess correctly
even with a binary guess.
In conclution, we want our data to be a snapshot of the real world. Of the same distribution (why we tend to select random samples out of a real-world distribution dataset),
in order to try to make our model as general and "close to the real world" as possible.

**3.True**
Any example our predictor sees in all stages of the training, including any splitting to test-validation,
will **always** be seperated from a test set we decide on at the beginning of the training process.
We split our training data into k equal folds, each time we use 1 as validation group and the other k-1 as train group.
In each fold, we will train the model on the train set, and test it agaisnt the validation set.
In this way we can fine-tune the hyper paramaters to find the best set of parameters we would want to use for the model.
After we are done, we will use all of our data and train a model with them, using the hyper parameters we have tuned beforehand. This marks the end of the cross-validation.
Now we can test our success against the test data group (Unrelated to cross-validation).

**4.True**
Just like the test data is never used and is hidden from our model, we use the validation set, a never-
 The performance of each fold on the validation set is then used in order to approximate the expected loss on unsampled data and thus serves as 
a proxy for the model's generalization error.
"""

part1_q2 = r"""
**The approach suggested is utterly and completely ridiculous...**

What the friend is doing is massivley overfitting his predictor.
He should try to tune his paramater while using k-fold CV or a similar method.
The friend is relying on his praticular samples too much, and instead of trying to create a general model
that will fit best to unseen data, he is overfitting in attempt to get the best results on a **specific** set of samples, training and validation set."""

# ==============
# Part 2 answers

part2_q1 = r"""
**Your answer:**

Lets look at the edge cases regarding this question:
For k = 1 we only look at the most nearest neighbor.
This can make an over-fit over our training set which is not
good for a generalized model. On the other hand, for k = samples_num
we will get the same answer for any sample, the class which appears
the most in the training set. For the best k we need to pick a k which
is in between 1 and samples_num and the exact point can change between
different module. In out case we found out the the best k is 3 from k_choices.

"""

part2_q2 = r"""
**Your answer:**

1. Using this suggestion will make our model over-fitting to the training set.
In addition, we will train and test the model on the same set. In CV
we will validate the model on a different and unseen test-set which can tell us
better if the model is good and create better results.

2. Using this suggestion will also our model over-fitting to the test set.
We want our model to be as generalized as possible since we dont know what
our real test-set will be. Therefore, k-fold is a better solution in which
that we validate our result on a different test-set and training-set each
time and make an average between different training-sets and test-sets to
generalize our model which will create better results.

"""

# ==============

# ==============
# Part 3 answers

part3_q1 = r"""
**Your answer:**

We say that $\Delta$ is arbitrary because as long as $\Delta > 0$ the score for the correct class is higher than
the other classes by $\Delta$ but the exact value is not important since the optimum will not change.
In addition, the two parameters $\Delta$ and $\lambda$ are related and for each $\Delta > 0$ we 
can find a $\lambda$ which will create a good results.

"""

part3_q2 = r"""
**Your answer:**

1. The linear model is learning an average representation of each digit. He is finding the best weights 
that will make the loss function minimal. When a test-sample is given the model calculate the score for each class 
using the weights and return the class y_pred with the highest score. For test-sample of a digit which have unclear
handwrite, the weights might be wrong because they only try to see how much is that sample is close to the average
representation of the digit. For example, for the digit 9 which is unclear and look a little bit like 4 the model
predicted 4 because the representation is closest to the average representation of 4 by the model weights.

2. KNN does not save any weights, but save the whole training set. In some sense, he does not learn anything but
save all of the learning materials. Then, he checks the labels of the k nearest example he has. It is not similar
to linear classifier which is learning the weights that represent the average case from all of the training set.
So linear classifier is looking for the average from all of the training set and KNN is looking for the most similar
samples he has in the training set and return the label which appear most times. It is similar in that eventually
both are trying to compare the test-sample to training-set samples (in average or k nearest).


"""

part3_q3 = r"""
**Your answer:**

1. For a learn_rate which it too low the decrease in loss will be very slow for the training and validation sets.
On the other hand, for a learn_rate which is too high the model can be over-fitting to the training-set and we will
see good results (low loss) on the training set but not on the validation set and the model can get close and 
far from the optimum and not be consistent in the loss graph. Our learn_rate is a good rate in the middle because
we see that both training-set and validation-set are decreasing in a pretty good pace and stabilize after few epochs. 

2. The model is slightly over-fitted to the training set because we can see that the loss on the training-set is
lower then the validation-set and decreasing in a higher pace. This means that the model is better for the
training set which indicates the he is slightly over-fitted to the training-set. The model is not highly over-fitted
because the validation-set loss is also decreasing and stabilizing at later epochs.

"""

# ==============

# ==============
# Part 4 answers

part4_q1 = r"""
If the points in a residual plot are randomly dispersed around the horizontal axis, a linear regression model is appropriate for the data;
otherwise, a nonlinear model is more appropriate. Therefore, the ideal pattern to see is randomly dispersed points around the horizontal line.
The top 5 features show errors on correlating with different patterns, none of them fit the description of a linear regression model, even though some of them better then others (RM fits better to a linear model then TAX for example).
The plot we recieved after the cross-validation is alot closer to a horizontal pattern, meaning that that praticular model fits alot better to a linear regression model then previous iterations.
"""

part4_q2 = r"""
1. **Yes**. 
A linear regression means that we model using a linear predictor function.
This does not change when we alter and transform the features we use as input to the model, and no changes occur to the way the modeling actually happens.
We use a linear regression model, but the model trains on a different dataset (features change) then our original since we alterd the features.
we still parametrize by weights vector and bias term $\vec{w}, b$, such that given a sample $\vec{x}$ our prediction is 

$$
\hat{y} = \vectr{w}\vec{x} + b.
$$

2. **False**
If we take as an example: $y=sqrt(|x|)+exp(x)$
There is no linear transformation that will make this relation linear.

3. The reason to use/transform non-linear features to linear features in a linear classifier model is that we use the linear features we **create** to craft a hyper-plane decision boundry using the crafted features,
even though for the original feature, it's parameter space could be anything imaginable, possibly alot more complex and could consist of a variaty of none-linear shapes.
This way even if the data cannot be linearly seperated in the original space, it could be seperable in the new hyperplane we have created.      
"""

part4_q3 = r"""
1. From the tests we ran, we could see that the differences in our results were when we were playing with the hyperparams are on a very small scale, for example, we could see
a large difference between 0.01 to 0.1. On the contrary, when we try to use large parameters, for example 100 or 1000 or even 0.8 and 1 or even within a small delta of 0.8 for insance, can not get much difference in our results.
We infer that our specific hyperparams act as a **logarithm**, and since we want to try numbers randomly on the logarithmic space we search on,
choosing a random uniformly from a range and then applying the logarithmic scale to it makes alot of sense. (Evenly distributed hyperparams samples along our logarithm scale)

2. **During training, we fitted the model 180 times**
degree = np.arange(1, 4) = 3
lambda_options = len(np.logspace(-3, 2, base=10, num=20)) = 20
Nfolds = 3
$Nfolds*degree*lambda=3*20*3=180$"""

# ==============
