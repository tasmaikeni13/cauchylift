# Mathematical Proof Audit

### Theorem 1: Degree-0 Scale Invariance
**Claim:** For any matrix $M \ne 0$ and scalar $\alpha > 0$, $U(\alpha M) = U(M)$.
**Proof:**
1. Row RMS scales as $\text{RMS}_{\text{row}, i}(\alpha M) = \sqrt{\frac{1}{n} \sum_j (\alpha M_{ij})^2} = \alpha \, \text{RMS}_{\text{row}, i}(M)$.
2. Column RMS scales as $\text{RMS}_{\text{col}, j}(\alpha M) = \alpha \, \text{RMS}_{\text{col}, j}(M)$.
3. Denominator scales as $D_{ij}(\alpha M) = \alpha D_{ij}(M)$.
4. Rational entry scales as $Z_{ij}(\alpha M) = \frac{\alpha M_{ij}}{\alpha D_{ij}(M)} = Z_{ij}(M)$.
5. Frobenius norm $\|Z(\alpha M)\|_F = \|Z(M)\|_F$.
6. Therefore, $U(\alpha M) = \rho \frac{Z(\alpha M)}{\|Z(\alpha M)\|_F} = \rho \frac{Z(M)}{\|Z(M)\|_F} = U(M)$. Q.E.D.

### Theorem 2: Coordinate Magnitude Bounds
**Claim:** For any active entry $(i, j)$, $|Z_{ij}| \le \min(\sqrt{n}, \sqrt{m})$.
**Proof:**
1. $D_{ij} = \text{RMS}_{\text{row}, i} + \text{RMS}_{\text{col}, j} \ge \text{RMS}_{\text{row}, i} = \sqrt{\frac{1}{n}\sum_k M_{ik}^2} \ge \frac{|M_{ij}|}{\sqrt{n}}$.
2. Similarly, $D_{ij} \ge \text{RMS}_{\text{col}, j} \ge \frac{|M_{ij}|}{\sqrt{m}}$.
3. Hence $D_{ij} \ge \max\left(\frac{|M_{ij}|}{\sqrt{n}}, \frac{|M_{ij}|}{\sqrt{m}}\right)$.
4. Taking reciprocals gives $|Z_{ij}| = \frac{|M_{ij}|}{D_{ij}} \le \min(\sqrt{n}, \sqrt{m})$. Q.E.D.

### Theorem 3: Strict Descent Alignment
**Claim:** For any non-zero matrix $M$, $\langle M, U(M) \rangle_F > 0$.
**Proof:**
1. $\langle M, U(M) \rangle_F = \frac{\rho}{\|Z\|_F} \sum_{i,j} M_{ij} Z_{ij} = \frac{\rho}{\|Z\|_F} \sum_{i,j, M_{ij} \ne 0} \frac{M_{ij}^2}{D_{ij}}$.
2. Since $D_{ij} > 0$ whenever $M_{ij} \ne 0$, every term in the sum is strictly positive ($M_{ij}^2 / D_{ij} > 0$).
3. Since $M \ne 0$, at least one term is non-zero, making the sum strictly positive. Q.E.D.
