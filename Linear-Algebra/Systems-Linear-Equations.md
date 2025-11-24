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

## Fundamental Concepts of Linear Algebra

This section focuses on the definitions of linear and non-linear equations and how systems of linear equations are understood geometrically and analytically using concepts like dependence and the determinant.

---

## 1. Linear vs. Non-Linear Equations

The key distinction lies in the degree of the variables and the operations performed on them.

### Linear Equations

* **Definition:** An equation is linear if every variable has a degree of $\mathbf{1}$ and the variables are only combined through addition or subtraction, scaled by numerical coefficients (scalars).
* **General Form:** $a_1x_1 + a_2x_2 + \dots + a_nx_n = c$
* **Example:** $3x + 2y - z = 10$

### Non-Linear Equations

* **Definition:** An equation is non-linear if it involves variables raised to a power other than one, or if it includes non-linear functions (like trigonometric, exponential, or logarithmic functions).
* **Examples of Non-Linear Terms:**
    * Variables with exponents: $a^2 + b^2 = 1$
    * Variables in a denominator (implying a negative exponent): $a/b$
    * Transcendental functions: $\sin(x)$, $\log(x)$, $e^x$

---

## 2. Geometric Interpretation of Linear Systems

**Linear Algebra** is the study of linear equations and their transformations.

### Visualizing Solutions

When we have a system of linear equations, a solution is a set of values for the variables that satisfy *all* equations simultaneously. Geometrically, this solution is the **intersection** of the shapes defined by the individual equations.

* **Two Variables (2D Plane):** A linear equation with two variables (e.g., $ax + by = c$) corresponds to a **line** in a two-dimensional plane. The solution to a system of two equations is the point where the two lines intersect. 

* **Three Variables (3D Space):** A linear equation with three variables (e.g., $ax + by + cz = d$) corresponds to a **plane** in three-dimensional space. The solution to a system of three equations is the single **point** where all three planes intersect. 

---

## 3. Linear Dependence and System Singularity

The nature of a system's solution (unique, infinite, or none) is determined entirely by the coefficients of the variables, not by the constant terms (the Right-Hand Side, or RHS) of the equations.

### Linear Dependence Between Rows (Equations)

* **Definition of Dependence:** Linear dependence exists between two or more equations (rows in the matrix) if one row can be obtained by multiplying another row by a scalar (or by adding multiples of other rows).
    * **Example:** If $a + b = 5$ and $2a + 2b = 10$, the second equation is simply $2 \times (\text{first equation})$. The rows are linearly dependent.
* **Definition of Independence:** Rows are **linearly independent** if no row can be expressed as a linear combination of the other rows.

### Non-Singular vs. Singular Systems

This dependence directly determines the system's nature and its ability to yield a unique solution:

| Property | Condition | Implication for Solution |
| :--- | :--- | :--- |
| **Non-Singular** (Complete) | The rows (equations) are **linearly independent**. | **Unique Solution** exists. |
| **Singular** (Redundant or Contradictory) | The rows (equations) are **linearly dependent**. | **Infinite Solutions** (redundant) or **No Solution** (contradictory). |

### The Determinant

The **determinant** is a scalar value calculated from the elements of a square matrix that summarizes these properties:

* **Non-Singular Matrices:** Have a **non-zero determinant** ($\det(A) \neq 0$). This guarantees the system has a unique solution and the matrix is invertible.
* **Singular Matrices:** Have a **zero determinant** ($\det(A) = 0$). This means the system is either redundant or contradictory (infinite or zero solutions).
