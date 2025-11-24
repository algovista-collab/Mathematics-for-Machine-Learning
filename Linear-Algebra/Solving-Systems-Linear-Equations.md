# Advanced Linear Algebra Concepts in Machine Learning

This section details key concepts related to solving linear systems, determining the amount of information in a matrix (Rank), and its application in data compression (SVD).

---

## 1. Gaussian Elimination and Row Echelon Form

### Gaussian Elimination
* **Definition:** **Gaussian elimination** is an algorithm used to solve systems of linear equations.
* **Process:** It uses a sequence of elementary row operations (swapping rows, multiplying a row by a non-zero scalar, or adding a multiple of one row to another) to transform the coefficient matrix into an upper triangular form. When we do the row operations and one or more rows are zeros and when the column of constants is also zero for the same row, then it has system of infinitely many solutions, otherwise is has no solution (contradictory).

### Row Echelon Form (REF)
* **Definition:** A matrix is in **Row Echelon Form (REF)** when:
    1. All non-zero rows are above any rows of all zeros.
    2. The leading entry (the first non-zero number) of a non-zero row, called the **pivot**, is in a column to the right of the leading entry of the row above it.
    3. All elements **below the leading entries (pivots)** are $\mathbf{0}$.
* **Note:** While it's common in the **Reduced Row Echelon Form (RREF)** to require the pivots to be 1, in standard REF, they can be any non-zero number. 

<img width="738" height="273" alt="Screenshot 2025-11-06 103835" src="https://github.com/user-attachments/assets/c2c7e7c0-1604-4d7d-a4a5-2439c622c1ff" />

---

## 2. Rank of a Matrix and Information Content

### Rank of a Matrix
* **Definition:** The **rank of a matrix** is the dimension of the column space (the maximum number of linearly independent column vectors) or, equivalently, the dimension of the row space (the maximum number of linearly independent row vectors).
* **Application:** The rank is related to the **amount of essential information** or "storage space" needed for the corresponding data (e.g., in image representation).
    * If a system of equations has 2 pieces of unique, non-redundant information, its rank is 2.
    * A matrix has **full rank** if its rank is equal to the number of rows (or columns, whichever is smaller), signifying it carries the **maximum amount of information possible**.

### Non-Singular Matrices and Full Rank
* **Condition:** A square matrix is **non-singular** (and therefore invertible) **if and only if** it has **full rank**, meaning the rank is equal to the number of rows ($\text{rank}(A) = m$) and the number of columns ($\text{rank}(A) = n$).

<img width="1167" height="455" alt="Screenshot 2025-11-06 104144" src="https://github.com/user-attachments/assets/4dacd586-3b54-4c95-8642-bbe0548764d3" />

<img width="1122" height="193" alt="Screenshot 2025-11-06 115418" src="https://github.com/user-attachments/assets/967fa3e1-c0cd-4311-a6bb-282a03478792" />

---

## 3. Rank and the Solution Space

The rank of the coefficient matrix is closely related to the dimension of the **solution space** of the homogeneous system.

### Solution Space (Null Space)
* **Definition:** The **solution space** (or **null space**) is the set of all solutions to the homogeneous system of equations $\mathbf{A}\mathbf{x} = \mathbf{0}$, where the constants on the RHS are all zero.
* **Interpretation:** The dimension of the solution space represents the **number of free variables** or the "size" of the set of solutions.

### The Rank-Nullity Theorem (Implicitly)
For a matrix $\mathbf{A}$ with $n$ columns (variables), the following relationship holds:

$$\text{rank}(\mathbf{A}) + \text{dimension of solution space} = n \quad (\text{number of columns})$$

<img width="1045" height="397" alt="image" src="https://github.com/user-attachments/assets/0c6891d3-d43b-4a83-b917-4b7a9a484fea" />

Assuming a $2 \times 2$ matrix ($\mathbf{A}$ has 2 rows and 2 columns, $n=2$):
* **Non-Singular (Full Rank):**
    * $\text{rank} = 2$.
    * Dimension of solution space $= 2 - 2 = \mathbf{0}$. (Only the trivial solution $\mathbf{x} = \mathbf{0}$ exists).
* **Singular (Redundant):**
    * $\text{rank} = 1$.
    * Dimension of solution space $= 2 - 1 = \mathbf{1}$. (Solutions form a line).
* **Maximum Dependence:**
    * $\text{rank} = 0$. (Matrix is all zeros, no information).
    * Dimension of solution space $= 2 - 0 = \mathbf{2}$. (All vectors in $\mathbb{R}^2$ are solutions).

---

## 4. Singular Value Decomposition (SVD)

* **Definition:** **Singular Value Decomposition (SVD)** is a powerful matrix factorization technique.
* **Application:** SVD can be used for **Low-Rank Approximation**—it can reduce the rank of a matrix while ensuring the resulting low-rank matrix is the "closest" possible approximation to the original matrix (in a least-squares sense).
* **Use in ML/DS:** This technique is crucial for **data compression** and **Dimensionality Reduction** (like Principal Component Analysis, PCA)
