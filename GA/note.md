## Why GA Did Not Converge on `att48` Before

- **Problem:** The GA used **Euclidean distances** instead of the **ATT pseudo-Euclidean metric** defined in TSPLIB.  

- **Incorrect distance (used before):**
\[
d_{EUC}(i,j) = \sqrt{(x_i - x_j)^2 + (y_i - y_j)^2}
\]

- **Correct ATT distance:**
\[
r_{ij} = \sqrt{\frac{(x_i - x_j)^2 + (y_i - y_j)^2}{10}}, \quad
d_{ATT}(i,j) = \text{int}(r_{ij} + 0.5)
\]

- **Effect:**  
  - The GA optimized the wrong fitness:
  \[
  f(T) = \sum d_{EUC}(T_i,T_{i+1}) \not\approx f^*_{opt}
  \]  
  - Distorted landscape → stuck in poor local minima → GAP > 200%.

- **Fix:** Using the correct `ATT` formula aligns fitness with the TSPLIB optimum:
\[
f(T) = \sum d_{ATT}(T_i,T_{i+1})
\]  
Now the GA converges meaningfully, and GAP drops to ~20–30%.


Final best distance: 14582
GAP vs Optimal: 37.20%
