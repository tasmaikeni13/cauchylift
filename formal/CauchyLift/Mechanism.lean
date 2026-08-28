import Mathlib

/-!
# Exact two-mode recurrence

On a two-coordinate diagonal quadratic, a direction with coordinate ratio
`q^3` and an exact line search leaves the new gradient orthogonal to that
direction. The resulting gradient ratio is `-q^-3`. This is the deterministic
algebraic core of the Phase 2 mode-alternation signature.
-/

namespace CauchyLift

theorem twoMode_cubic_ratio
    {q next₁ next₂ : ℝ}
    (hq : q ≠ 0) (hnext₂ : next₂ ≠ 0)
    (horthogonal : q ^ 3 * next₁ + next₂ = 0) :
    next₁ / next₂ = -1 / q ^ 3 := by
  field_simp
  nlinarith

theorem twoMode_normalized_gradient_ratio
    {q next₁ next₂ : ℝ}
    (hq : q ≠ 0) (hnext₂ : next₂ ≠ 0)
    (horthogonal : q * next₁ + next₂ = 0) :
    next₁ / next₂ = -1 / q := by
  field_simp
  nlinarith

theorem twoMode_two_step_amplification {q : ℝ} :
    -1 / (-1 / q ^ 3) ^ 3 = q ^ 9 := by
  field_simp
  ring

end CauchyLift
