# Mathematics for Machine Learning: Linear Algebra and Calculus

This document summarizes the core applications of **Linear Algebra** (specifically **Matrices** and **Systems of Linear Equations**) and **Calculus** in the context of Machine Learning and Data Science.

---

## Linear Algebra: Representing Data and Solving Systems

### Matrices

* **Role:** Matrices are the fundamental building blocks used to represent data and its transformations in Machine Learning and Data Science.
    * A dataset with $m$ samples and $n$ features is typically stored as an $m \times n$ matrix.

### Systems of Linear Equations

A common problem in machine learning (like **Linear Regression**) is finding a set of weights ($\mathbf{w}$) and a bias ($b$) that best fit a series of linear equations, one for each data sample.

The system of equations is:

$$
\begin{aligned}
w_1x_1^{(1)} + w_2x_2^{(1)} + \dots + w_nx_n^{(1)} + b &= y^{(1)} \\
w_1x_1^{(2)} + w_2x_2^{(2)} + \dots + w_nx_n^{(2)} + b &= y^{(2)} \\
&\dots \\
w_1x_1^{(m)} + w_2x_2^{(m)} + \dots + w_nx_n^{(m)} + b &= y^{(m)}
\end{aligned}
$$

We aim to find the values for the weight vector $\mathbf{w} = [w_1, w_2, \dots, w_n]$ and the bias $b$ that best satisfy all $m$ equations simultaneously.

#### Matrix Notation

This system is compactly represented using linear algebra as:

$$\mathbf{W} \cdot \mathbf{X} + \mathbf{b} = \mathbf{\hat{y}}$$

* **$\mathbf{W}$ (Weights):** A vector or matrix of parameters we are trying to determine.
* **$\mathbf{X}$ (Features):** The data matrix containing the input samples.
* **$\mathbf{b}$ (Bias):** A scalar or vector added to the result.
* **$\mathbf{\hat{y}}$ (Predicted Output):** The output predicted by the model.

**Linear Algebra's Role:** With linear algebra techniques, we solve this system of equations **empirically/iteratively** (e.g., via gradient descent, which uses calculus) or **analytically** (e.g., using the Normal Equation) to find the **best-fit linear solution** to the system.

### Properties of Linear Systems

The nature of the solution depends on the properties of the system's underlying matrix (the feature matrix $\mathbf{X}$).

| System Type | Description | Solution | Example |
| :--- | :--- | :--- | :--- |
| **Non-Singular** (Complete) | Contains sufficient, non-contradictory, non-redundant information. Carries **more information** and is **more useful**. | **Unique Solution** | $a + 3b = 13$  and $4a + b = 15$ |
| **Redundant** (Singular) | Contains unnecessary, dependent information. | **Infinitely Many Solutions** | $a + b = 5$ and $2a + 2b = 10$ |
| **Contradictory** (Singular) | Contains information that conflicts with other information. | **No Solution** | $a + b = 10$ and $2a + 2b = 12$ |

---

## 📈 Calculus: Optimization and Learning

* **Role:** Calculus is used for **maximizing and minimizing functions**, which is crucial for the **learning** process in machine learning.
* **Application:** It is used for **minimizing the loss function** (or maximizing the cost function/likelihood, depending on the context).
    * To train a model, we define a **Loss Function** (e.g., Mean Squared Error) that quantifies how poorly the model's predictions ($\mathbf{\hat{y}}$) match the true values ($\mathbf{y}$).
    * **Gradient Descent**, the primary optimization algorithm, uses the **derivative** (from calculus) to find the direction of steepest descent, iteratively adjusting the weights ($\mathbf{w}$) and bias ($b$) to find the minimum of the loss function.

<img width="1027" height="255" alt="image" src="https://github.com/user-attachments/assets/9aee7adb-eb3d-4f73-b43a-53b79b4841c9" />

