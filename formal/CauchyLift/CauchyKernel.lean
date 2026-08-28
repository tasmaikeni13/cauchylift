import Mathlib

/-!
# Two-by-two Cauchy-kernel identity

This checks the smallest nontrivial rank-lift certificate.  The paper's general
rank statement additionally requires every entry of both rank-one factors to
be nonzero, because the outer diagonal factors must preserve rank.  The general
Cauchy determinant product formula and those diagonal-matrix rank steps are not
claimed as fully formalized here.
-/

namespace CauchyLift

theorem rankOne_energy_factor {U V u v : ℝ}
    (hU : U ≠ 0) (hV : V ≠ 0) :
    2 * (U * V) - u ^ 2 * V - v ^ 2 * U =
      U * V * ((1 - u ^ 2 / U) + (1 - v ^ 2 / V)) := by
  field_simp
  ring

theorem cauchy_two_by_two_determinant
    {x₁ x₂ y₁ y₂ : ℝ}
    (h11 : x₁ + y₁ ≠ 0) (h12 : x₁ + y₂ ≠ 0)
    (h21 : x₂ + y₁ ≠ 0) (h22 : x₂ + y₂ ≠ 0) :
    1 / (x₁ + y₁) * (1 / (x₂ + y₂)) -
        1 / (x₁ + y₂) * (1 / (x₂ + y₁)) =
      ((x₁ - x₂) * (y₁ - y₂)) /
        ((x₁ + y₁) * (x₁ + y₂) * (x₂ + y₁) * (x₂ + y₂)) := by
  field_simp
  ring

theorem cauchy_two_by_two_nondegenerate
    {x₁ x₂ y₁ y₂ : ℝ}
    (hx : x₁ ≠ x₂) (hy : y₁ ≠ y₂)
    (h11 : x₁ + y₁ ≠ 0) (h12 : x₁ + y₂ ≠ 0)
    (h21 : x₂ + y₁ ≠ 0) (h22 : x₂ + y₂ ≠ 0) :
    1 / (x₁ + y₁) * (1 / (x₂ + y₂)) -
        1 / (x₁ + y₂) * (1 / (x₂ + y₁)) ≠ 0 := by
  rw [cauchy_two_by_two_determinant h11 h12 h21 h22]
  apply div_ne_zero
  · exact mul_ne_zero (sub_ne_zero.mpr hx) (sub_ne_zero.mpr hy)
  · exact mul_ne_zero
      (mul_ne_zero (mul_ne_zero h11 h12) h21) h22

end CauchyLift
