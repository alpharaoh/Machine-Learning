
## Support Vector Machines (SVM) - Under the Hood
In this markdown textbook, we wil use the convention of $b$ being the bias term and the wieghts vector will be called __w__. No bias feature will be added to the input feature vectors.
### Decision Functions and predictions

The linear SVM classifier model predicts the class of the new instance __x__ by simply computing the decision function __w__$^T$__x__ + $b$ = $w_1x_1 + ... + w_nx_n+b$. If the result is positive, the predicted class $\hat{y}$ is the positive class (1), and other wise it is the negative class (0), see equation below.

$\hat{y} = 0$ if $w^Tx+b < 0$
$\hat{y} = 1$ if $w^Tx+b \ge 0$

Figure 1 below shows the decision function that corresponds to the model in the right in the Figure 2. It is a 2D plane because this dataset has two features (petal width and petal length). The decision boundary is the set of points where the decision function is equal to 0: it is the intersection of two planes, which is a straight line (represented by solid line)

Figure 2.
![[hyperparam.png]]

Figure 1. Decision function for the iris dataset
![[underhood1.png]]

The dashed lines represent the points where the deision function equals 1 or -1. Training a linear SVM classifier means finding the values of __w__ and $b$ that make this margin as wide as possible while avoiding margin violations (hard margin) or limiting them (soft margin)

## Training objective

Consider the slope of the decision function: it's equal to the norm of the weight vector, ||__w__||. If we divide this slope by 2, the points where the deciison function is equal to $\pm$ 1 

// Will go back to SVM Mathamatics later on, starting decision trees.