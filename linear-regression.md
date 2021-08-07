## Linear Regression



 **What is Linear Regression?**
    
    Linear regression is the next step up after correlation. It is used when we want to predict the value of a variable
    based on the value of another variable. The variable we want to predict is called the dependent variable (or sometimes,
    the outcome variable).
    
   
    
  **How do you calculate simple linear regression?**
    
    For the equation of the form $`Y= \beta_{0} + \beta_{1} X`$, where $`Y`$ is the dependent variable (that's the variable that goes
    on the $`Y`$ axis), $`X`$ is the independent variable (i.e. it is plotted on the $`X`$ axis), $\beta_{1}$ is the slope of 
    the line and a is the y-intercept.
    
   
    
    
   **What is linear and non linear regression?**
    
    A linear regression equation simply sums the terms. While the model must be linear in the parameters. You can raise an
    independent variable by an exponent to fit a curve. For instance, you can include a squared or cubed term. Nonlinear 
    regression models are anything that doesn't follow this one form.
    
    
    
   **What is the difference between regression and correlation?**
    
    The difference between these two statistical measurements is that correlation measures the degree of a relationship between
    two variables (x and y), whereas regression is how one variable affects another.
    
    Simple Linear regression can be represented as:
    
   [Figure]
    
    The task is to determine the coefficients $\beta_{0}$ and $\beta_{1}$ whereas Correlation coefficient is defined as:
    
    [Figure]
    
    For example different values of correlation coefficient, which renges from -1 to +1 can be demonstrated as: 
    
    [Figure]
    
    
    - What are regression coefficients? How do you determine them analytically?
    
    Determining regression coefficient can be carried out in analytic manner which needs cost function defined, (based on equation 1), which is 
    
    \begin{align}
       Q(\beta_{0},\beta_{1}) =  \sum_{i}^{N}\epsilon_{i} = \sum_{i}^{N} (Y_i - \hat{Y}) = \sum_{i}^{N} (Y_i - \beta_{0} -\beta_{1} X_{i})
    \end{align}
    
    The optimization problem can be set up as 
    
    \begin{equation}
        min_{\beta_{0},\beta_{1}} Q(\beta_{0},\beta_{1})
    \end{equation}
    
    The minimization of this cost function leads to the solution in the form
    \begin{equation}
        \hat{\beta}_{0} = \bar{Y} - \hat{\beta}_{1} \bar{x}\\
    \end{equation}
    
    \begin{align}
        \hat{\beta}_{1} = \frac{ \sum_{i}^{N} (x_{i} - \bar{x}) \sum_{i}^{N} (y_{i} - \bar{y})} {(x_i - \bar{x})} = \frac{s_{xy}}{s_xs_y} = r_{xy}\frac{s_x}{s_y}
    \end{align}
    
    
    - How do you obtain closed form solution in linear regression?
    
    Lets consider a generalized form of linear regression in terms of vector and matrices.
    
    \begin{equation}
        \hat{Y} = X^{T}\beta = x_{1}\beta_{1} +x_{2}\beta_{2} + ... +x_{p}\beta_{p}
    \end{equation}
    
    where $x_{1}, x_{2}, ..., x_{p}$ are feature vectors and $\beta$ is parameter vector with $\beta_{1}, \beta_{2},..., \beta_{p} $ as elements. This leads to cost function
    
    \begin{equation}
        J(\beta) = \frac{1}{2n} (X\beta - Y)^{T}(X\beta -Y)
    \end{equation}
    
    Taking first derivative w.r.t. $\beta$ to zero (assuming $X^{T}X$ is positive definite) gives
    
    \begin{equation}
        X^{T}(X\beta - Y) = 0
    \end{equation}
    
    Which in rearrangement gives
    
    \begin{equation}
        \hat{\beta} = (X^{T}X)^{-1}X^{T}Y
    \end{equation}
    
     %----------------------------------------------------------
    
    \item {What is OLS regression model?}
    
    In statistics, ordinary least squares (OLS) is a type of linear least squares method for estimating the unknown parameters in a linear regression model. 
    
    Under these conditions, the method of OLS provides minimum-variance, mean-unbiased estimation when the errors have finite variances.
    
     %----------------------------------------------------------
    
    \item {What is difference between linear and logistic regression?}
    
    Linear regression is used for predicting the continuous dependent variable using a given set of independent features whereas Logistic Regression is used to predict the categorical. Linear regression is used to solve regression problems whereas logistic regression is used to solve classification problems.
    
     %----------------------------------------------------------
    
    \item {What is a regression curve?}
    
    A curve that best fits particular data according to some principle (as the principle of least squares)
    
     %----------------------------------------------------------
    
    \item {What is the difference between simple linear regression and multiple linear regression?}
    
     Simple linear regression has only one x and one y variable. Multiple linear regression has one y and two or more x variables. For instance, when we predict rent based on square feet alone that is simple linear regression.
     
      %----------------------------------------------------------
     
     \item {How do you tell if a regression model is a good fit?}
     
    In general, a model fits the data well if the differences between the observed values and the model's predicted values are small and unbiased. Before you look at the statistical measures for goodness-of-fit, you should check the residual plots.
    
     %----------------------------------------------------------
    
    
    \item {What is a good R squared value?}
    
    Any study that attempts to predict human behavior will tend to have R-squared values less than 50 percent. However, if you analyze a physical process and have very good measurements, you might expect R-squared values over 90
    
     %----------------------------------------------------------
    
    \item {What does R Squared mean?}
    
    coefficient of determination
    R-squared (R2) is a statistical measure that represents the proportion of the variance for a dependent variable that's explained by an independent variable or variables in a regression model. ... It may also be known as the coefficient of determination.
    
    How do you interpret R Squared examples?
    The most common interpretation of r-squared is how well the regression model fits the observed data. For example, an r-squared of 60 pt reveals that 60 pt of the data fit the regression model. Generally, a higher r-squared indicates a better fit for the model.
    
     %----------------------------------------------------------
    
    \item {What is the difference between linear and polynomial regression?}
    
    Polynomial Regression is a one of the types of linear regression in which the relationship between the independent variable x and dependent variable y is modeled as an nth degree polynomial. 
    
    Polynomial Regression provides the best approximation of the relationship between the dependent and independent variable.
    
     %----------------------------------------------------------
    
    \item {What does the Optimizer usually do in gradient descent in linear regression?}
    
    Gradient descent is an optimization algorithm used to find the values of parameters (coefficients) of a function (f) that minimizes a cost function (cost). 
    
     %----------------------------------------------------------
     
     \item {How do you set up optimization problem and solve it in case of linear regression?}
     
     Consider the cost function created from mean squared error (why 2 in denominator?.
     
     \begin{equation}
     J(\theta) = \frac{1}{2n} \sum_{i}^{n}|h_{\theta}(x^{i}) - y^{i}|^{2}
     \end{equation}
    Where,
    \begin{equation}
        h_{\theta}(x^{i}) = \sum_{k}^{d} \theta_{k}x_{k}^{i}
     \end{equation}
     
     is predicted value for data $i$ of dimension (no of features) $d$. In optimization called gradient descent, one need to calculate the derivative of the cost function w.r.t. the parameters $\theta$ to update the parameter as
    
    \begin{equation}
        \theta \leftarrow \theta - \alpha \frac{\partial }{\partial \theta} J(\theta)
    \end{equation}
    with vector equation,
    \begin{equation}
      \frac{\partial }{\partial \theta} J(\theta)  = \frac{1}{n} \sum_{i}^{n}(h_{\theta}(x^{i}) - y^i)x^{i}
    \end{equation}
    
    Note: in terms of individual $\theta_j$
    
    \begin{equation}
      \frac{\partial }{\partial \theta_{j}} J(\theta)  = \frac{1}{n} \sum_{i}^{n}(\sum_{k}^{d}\theta_{k} x_{k}^{i} - y^i)x_{j}^{i}
    \end{equation}
    
    the simultaneous update of parameters $\theta_k, k =1,2,3...d $ becomes
    
    \begin{equation}
        \theta \leftarrow \theta - \frac{\alpha}{n} \sum_{i}^{n} (h_{\theta}(x^{i}) - y^i)x^{i}
    \end{equation}
     
    %----------------------------------------------------------
    
    \item {What is difference between closed form solution and gradient descent solution?}
    
    \begin{itemize}
        \item GD required multiple iteration but CFS is non iterative
        \item In GD one need to choose learning rate $\alpha$ nut in CFS it is not required
        \item GD works well with large N (data points) or large d (features) but CFS grows in complexity roughly $(O(n^3))$ for calculating $(X^{T}X)^{-1}$.
        \item GD can support incremental online learning but CFS need all data at same time.
    \end{itemize}
    
    
    %----------------------------------------------------------
    \item {What are the issues with Gradient Descent?}
    
    \begin{itemize}
        \item Converge to local optimum: restart from multiple starting points
        \item Only works with differentiable loss functions
        \item Problem of smaller or larger gradients: perform feature scaling of the data
        \item Tune learning rate: can use line search to find optimum learning rate
    \end{itemize}
    
    %----------------------------------------------------------
    
    \item {Explain Bias Variance in Linear Regression. What does it mean by bias-variance trade off?}
    
    In statistics, there are two critical characteristics of estimators to be considered: the bias and the variance. The bias is the difference between the true population parameter and the expected estimator:
    
    \begin{equation}
        Bias(\hat{\beta}_{OLS}) = E(\hat{\beta}_{OLS}) -\beta
    \end{equation}
    
    It measures the accuracy of the estimates. Variance, on the other hand, measures the spread, or uncertainty, in these estimates. It is given by
    
    \begin{equation}
        Var(\hat{\beta}_{OLS}) = \sigma^{2}(X^{T}X)^{-1}
    \end{equation}
    where the unknown error variance $\sigma^{2}$ can be estimated from the residuals as $\sigma^{2} = \frac{e^{T}e}{n-m}$ where $e = y - \hat{\beta}X$.
    
    The bias and variance is statistical context can be understood in the figure
    
    \begin{figure}[h!]
    \begin{center}
    \includegraphics[scale=0.9]{image/ml/lr/bias-variance-02.jpg}
    \caption{Bias Variance statistical meaning}
    \end{center}
    \end{figure}
    
    In the context of machine learning (i.e. bias-variance trade off), we prefer to observe the effect of bias and variance in the training accuracy where total error is a sum of bias and variance term.
    
    \begin{figure}[h!]
    \begin{center}
    \includegraphics[scale=0.5]{image/ml/lr/bias-variance-01.png}
    \caption{Bias Variance Trade off}
    \end{center}
    \end{figure}
    
    %---------------------------------------------------------
    
    \item {Derive MSE in terms of Bias and Variance of estimator over different realization of training data and one set of test data.}
    
    Consider a linear regression model is represented by  
    
    \begin{equation}
        y = f(x) + \epsilon
    \end{equation}
    Where $\epsilon$ is the noise with 0 mean and variance $\sigma_{\epsilon}$.
    \begin{equation}
        \mathbf{E}[\epsilon] = 0;~~ var(\epsilon) = \mathbf{E}[\epsilon^{2}] = \sigma_{\epsilon}^{2}
    \end{equation}
    Mean square Error is given by 
    \begin{equation}
        MSE = \mathbf{E}[(y - \hat{f}(x))^{2}]
    \end{equation}
    Bias is defined as the difference of the average value of prediction (over different realizations of training data) to the true underlying function f(x) for a given unseen (test) point x.
    \begin{equation}
        bias(\hat{f}(x)) = \mathbf{E}[\hat{f}(x)] -f(x))
    \end{equation}
    Variance is defined as the mean squared deviation of $\hat{f}(x)$ from its expected value $\mathbf{E}[\hat{f}(x)]$ over different realizations of training data.
    \begin{equation}
        var(\hat{f}(x)) = \mathbf{E}[(\hat{f}(x) - \mathbf{E}[\hat{f}(x)])^{2}]
    \end{equation}
    One can express MSE in terms of bias and variance terms as \cite{bias-variance-01}
    
    \begin{figure}[h!]
    \begin{center}
    \includegraphics[scale=0.5]{image/ml/lr/bv01.png}
    \end{center}
    \end{figure}
    
    The first expectation is over the distribution of unseen (test) points x, while the second over the distribution of training data, or over $\hat{f}(x)$ since $\hat{f}(x)$ depends on training data. If we were to write the above formula more explicitly, it would be:
    
    \begin{figure}[h!]
    \begin{center}
    \includegraphics[scale=0.5]{image/ml/lr/bv02.png}
    \end{center}
    \end{figure}
    
    
    %---------------------------------------------------------
    
    \item {Is linear regression a neural network?}
    
    Linear Network/Regression = Neural Network ( with No hidden layer) only input and output layer.
    
    A Neural network can be used as a universal approximator, so it can definitely implement a linear regression algorithm.
    
    As such, linear regression was developed in the field of statistics and is studied as a model for understanding the relationship between input and output numerical variables, but has been borrowed by machine learning. It is both a statistical algorithm and a machine learning algorithm
    
     %----------------------------------------------------------
    
    \item {Is linear regression always convex?}
    
    The Least Squares cost function for linear regression is always convex regardless of the input dataset, hence we can easily apply first or second order methods to minimize it.
    
     %----------------------------------------------------------
    
    \item {Which algorithm is used for regression?}
    
    Some of the popular types of regression algorithms are linear regression, regression trees, lasso regression and multivariate regression.

    Top six : Simple Linear Regression model, Lasso Regression, Logistic, regression, Support Vector Machines, Multivariate Regression algorithm, Multiple Regression Algorithm.
    
     %----------------------------------------------------------
    
    \item {What is Regularization in Linear Regression?}
    
    There are three popular regularization techniques, each of them aiming at decreasing the size of the coefficients: Ridge Regression, which penalizes sum of squared coefficients (L2 penalty). Lasso Regression, which penalizes the sum of absolute values of the coefficients (L1 penalty). Elastic Net.
    
     %----------------------------------------------------------
    
    \item {Why does regularization reduce Overfitting?}
    
    That's the set of parameters. In short, Regularization in machine learning is the process of regularizing the parameters that constrain, regularizes, or shrinks the coefficient estimates towards zero. In other words, this technique discourages learning a more complex or flexible model, avoiding the risk of Overfitting
    
     %----------------------------------------------------------
    
    \item {What is l1 and l2 regularization?}
    
    A regression model that uses L1 regularization technique is called Lasso Regression and model which uses L2 is called Ridge Regression. The key difference between these two is the penalty term. Ridge regression adds “squared magnitude” of coefficient as penalty term to the loss function.
    
     %----------------------------------------------------------
    
    \item {What is ridge regression used for?}
    
    Ridge Regression is a technique for analyzing multiple regression data that suffer from multicollinearity. When multicollinearity occurs, least squares estimates are unbiased, but their variances are large so they may be far from the true value.
    
     %----------------------------------------------------------
     
     \item{What is closed form solution of Ridge Regression?}
    
     Lets consider a generalized form of linear regression in terms of vector and matrices.
    
    \begin{equation}
        \hat{Y} = X^{T}\beta = x_{1}\beta_{1} +x_{2}\beta_{2} + ... +x_{p}\beta_{p} 
    \end{equation}
    
    where $x_{1}, x_{2}, ..., x_{p}$ are feature vectors and $\beta$ is parameter vector with $\beta_{1}, \beta_{2},..., \beta_{p} $ as elements. This leads to cost function
    
    \begin{equation}
        J(\beta) = \frac{1}{2n} (X\beta - Y)^{T}(X\beta -Y) + \frac{1}{2} \lambda |\beta|^{2}
    \end{equation}
    
    Taking first derivative w.r.t. $\beta$ to zero (assuming $X^{T}X$ is positive definite) gives
    
    \begin{equation}
        X^{T}(X\beta - Y) + \lambda \beta = 0
    \end{equation}
    
    Which in rearrangement gives
    
    \begin{equation}
        \hat\beta_{ridge} = (X^{T}X+\lambda I)^{-1}(X^{T}Y)
    \end{equation}
    
    %----------------------------------------------------------
    \item {What are upper and lower limit of $\lambda$ in Ridge regression?}
    
    The $\lambda$ parameter is the regularization penalty. 
    
    \begin{figure}[h!]
    \begin{center}
    \includegraphics[scale=0.7]{image/ml/lr/ridge02.png}
    \caption{Ridge coefficient selection. Source: Manning Publication}
    \end{center}
    \end{figure}
    
    \begin{itemize}
        \item In the lower limit as $\lambda \rightarrow 0$; $\hat{\beta}_{ridge} \rightarrow \hat{\beta}_{OLS}$
        \item In the upper limit as $\lambda \rightarrow \infty$; $\hat{\beta}_{ridge} \rightarrow 0$
    \end{itemize}
    
    %---------------------------------------------------------
     
    \item {What is cost function in Ridge regression? How do you define optimization problem?}
    
    The cost function in Ridge regression is expressed as
    
     \begin{equation}
     J(\theta) = \frac{1}{2n} \sum_{i}^{n}|h_{\theta}(x^{i}) - y^{i}|^{2} + \frac{\lambda}{2} |\theta|^{2}
     \end{equation}
     
     Where,
    \begin{equation}
        h_{\theta}(x^{i}) = \sum_{k}^{d} \theta_{k}x_{k}^{i}
     \end{equation}
    is predicted value for data $i$ of dimension (no of features) $d$.
    
    The optimization problem is defined as
    \begin{equation}
    min_{(\theta_{0},\theta_{i})}~~J(\theta_{0},\theta_{i})
    \text{subjected to} |\theta|^{2}\leq \epsilon 
    \end{equation}
 
    %----------------------------------------------------------
    \item {How do you apply gradient descent in case of Ridge Regression?}
    
    In optimization called gradient descent, one need to calculate the derivative of the cost function w.r.t. the parameters $\theta$ to update the parameter as
    
    \begin{equation}
        \theta \leftarrow \theta - \alpha \frac{\partial }{\partial \theta} J(\theta)
    \end{equation}

    with vector equation,
    \begin{equation}
      \frac{\partial }{\partial \theta_{0}} J(\theta)  = \frac{1}{n} \sum_{i}^{n}(h_{\theta}(x^{i}) - y^i)
    \end{equation}
    \begin{equation}
      \frac{\partial }{\partial \theta_{k}} J(\theta)  = \frac{1}{n} \sum_{i}^{n}(h_{\theta}(x^{i}) - y^i)x_{k}^{i} + \lambda \theta_{j}
    \end{equation}
    
    the simultaneous update of parameters $\theta_k, k =1,2,3...d $ becomes
    
    \begin{equation}
        \theta_{0} \leftarrow \theta_{0} - \frac{\alpha}{n} \sum_{i}^{n} (h_{\theta}(x^{i}) - y^i)
    \end{equation}
    
    \begin{equation}
        \theta_{k} \leftarrow \theta_{k} - \frac{\alpha}{n} \sum_{i}^{n} (h_{\theta}(x^{i}) - y^i)x_{k}^{i} -\alpha \lambda \theta_{k}
    \end{equation}
    
    
    %----------------------------------------------------------
    \item {What happens to the regression coefficients (weights) when you increase $\alpha$ in Ridge regression? }
    
    By decreasing the value of $\alpha$ it reduce the contribution of weights.
    
    \begin{figure}[h!]
    \begin{center}
    \includegraphics[scale=0.6]{image/ml/lr/ridge01.png}
    \end{center}
    \end{figure}
    
    %----------------------------------------------------------
    
    
    \item {Why is it called ridge regression?}
    
    Ridge regression adds a ridge parameter ($\lambda$), of the identity matrix to the cross product matrix, forming a new matrix $(X^{T}X + \lambda I)$. It's called ridge regression because the diagonal of ones in the correlation matrix can be described as a ridge.
    
    %----------------------------------------------------------
    \item {What are bias and variance terms in Ridge regression?}
    
    Incorporating the regularization coefficient in the formulas for bias and variance gives us

    \begin{figure}[h!]
    \begin{center}
    \includegraphics[scale=0.6]{image/ml/lr/bv04.png}
    \end{center}
    \end{figure}
    
    From there you can see that as $\lambda$ becomes larger, the variance decreases, and the bias increases. This poses the question: how much bias are we willing to accept in order to decrease the variance? Or: what is the optimal value for $\lambda$?
     
     %---------------------------------------------------------
    
    \item {Why does ridge regression reduce variance?}
    
    Ridge regression has an additional factor called $\lambda$ which is called the penalty factor which is added while estimating beta coefficients. This penalty factor penalizes high value of beta which in turn shrinks beta coefficients thereby reducing the mean squared error and predicted error.
    
     %----------------------------------------------------------
    
    \item {What is penalty in ridge regression?}
    
    Ridge regression shrinks the regression coefficients, so that variables, with minor contribution to the outcome, have their coefficients close to zero. The shrinkage of the coefficients is achieved by penalizing the regression model with a penalty term called L2-norm, which is the sum of the squared coefficients.
    
     %----------------------------------------------------------
    
    \item {How do you pick a lambda in Ridge Regression?}
    
    The value of lambda will be chosen by cross-validation. The plot shows cross-validated mean squared error. 
    
    \begin{figure}[h!]
    \begin{center}
    \includegraphics[scale=0.5]{image/ml/lr/ridge03.png}
    \end{center}
    \end{figure}
    
    As lambda decreases, the mean squared error decreases. Ridge includes all the variables in the model and the value of lambda selected is indicated by the vertical lines.
    
    
     %----------------------------------------------------------
    
    \item {Why does the lasso give zero coefficients?}
    
    \begin{figure}[h!]
    \begin{center}
    \includegraphics[scale=0.7]{image/ml/lr/lasso05.png}
    \caption{Selecting Lasso coefficients. Source Manning Publication}
    \end{center}
    \end{figure}
    
    The lasso performs shrinkage so that there are "corners" in the constraint, which in two dimensions corresponds to a diamond. If the sum of squares "hits" one of these corners, then the coefficient corresponding to the axis is shrunk to zero.
    
     %----------------------------------------------------------
    
    \item {How does Lasso regression work?}
    
    The LASSO imposes a constraint on the sum of the absolute values of the model parameters, where the sum has a specified constant as an upper bound. 
    
    \begin{figure}[h!]
    \begin{center}
    \includegraphics[scale=0.5]{image/ml/lr/lasso01.png}
    \end{center}
    \end{figure}
    
    This constraint causes regression coefficients for some variables to shrink towards zero. This is the shrinkage process.
    
     %----------------------------------------------------------
    
    \item {Is elastic net better than Lasso?}
    
    \begin{figure}[h!]
    \begin{center}
    \includegraphics[scale=0.6]{image/ml/lr/ridge-lasso-01.png}
    \caption{Ridge-lasso comparison: Source Manning Publication}
    \end{center}
    \end{figure}
    
    Lasso will eliminate many features, and reduce overfitting in your linear model. 
    
    Elastic Net combines feature elimination from Lasso and feature coefficient reduction from the Ridge model to improve your model's predictions.
    
    
     %----------------------------------------------------------
     
    \item {Is Lasso regression convex?}
    
    Convexity Both the sum of squares and the lasso penalty are convex, and so is the lasso loss function.
    However, the lasso loss function is not strictly convex. Consequently, there may be multiple $\beta$'s that minimize the lasso loss function.
    
     %----------------------------------------------------------
    
    \item {What is adaptive lasso?}
    
    The lasso is a popular technique for simultaneous estimation and variable selection (called ...).
    
    We then propose a new version of the lasso, called the adaptive lasso, where adaptive weights are used for penalizing different coefficients in the l1 penalty.
    
     %----------------------------------------------------------
    
    \item {What is elastic net in machine learning?}
    
    Elastic net is a popular type of regularized linear regression that combines two popular penalties, specifically the L1 and L2 penalty functions. 
    
    \begin{figure}[h!]
    \begin{center}
    \includegraphics[scale=0.5]{image/ml/lr/elastic01.png}
    \end{center}
    \end{figure}
    
    Elastic Net is an extension of linear regression that adds regularization penalties to the loss function during training
    
    %-------------------------------------------------
    
    \item {What is difference between Ridge, Lasso and Elastic Net?}
    
    Altogether
    \begin{figure}[h!]
    \begin{center}
    \includegraphics[scale=0.6]{image/ml/lr/elastic02.png}
    \caption{Ridge-lasso-Elastic net comparison: Source Manning Publication}
    \end{center}
    \end{figure}
    And separate
    \begin{figure}[h!]
    \begin{center}
    \includegraphics[scale=0.4]{image/ml/lr/elastic03.png}
    \caption{Ridge-lasso-Elastic net comparison: Source Manning Publication}
    \end{center}
    \end{figure}
    
    %----------------------------------------------------
    
    \item{ Can you apply normal gradient descent in case of lasso?}
    
    
    %----------------------------------------------------
    
    \item {When does elastic net reduce back to lasso and ridge regression?}
    
\end{enumerate}
