
## Support Vector Machines (SVM) - Under the Hood
In this markdown textbook, we wil use the convention of $b$ being the bias term and the wieghts vector will be called __w__. No bias feature will be added to the input feature vectors.
### Decision Functions and predictions

The linear SVM classifier model predicts the class of the new instance __x__ by simply computing the decision function __w__$^T$__x__ + $b$ = $w_1x_1 + ... + w_nx_n+b$. If the result is positive, the predicted class $\hat{y}$ is the positive class (1), and other wise it is the negative class (0), see equation below.

$\hat{y} = 0$ if $w^Tx+b < 0$
$\hat{y} = 1$ if $w^Tx+b \ge 0$

The figure below shows the decision function that corresponds to the model in the right in the equation. It is a 2D plane because this dataset has two features (petal width and petal length). The decision boundary is the set of points where the decision function is equal to 0: it is the intersection of two planes, which is a straight line (represented by solid line)