import Mathlib

/-!
# Scalar core of the CauchyLift proof

The matrix argument in the paper reduces to normalized nonnegative cell
energies.  This file checks the complement inequalities and the closing
alignment algebra without encoding a GPU implementation.
-/

namespace CauchyLift

open scoped BigOperators

theorem cotransverse_nonnegative {S r c : ℝ} (hr : r ≤ S) (hc : c ≤ S) :
    0 ≤ (S - r) + (S - c) := by
  linarith

theorem normalized_cotransverse_lower {row column cell : ℝ}
    (hUnion : row + column - cell ≤ 1) :
    1 - cell ≤ 2 - row - column := by
  linarith

theorem normalized_cotransverse_upper {row column : ℝ}
    (hrow : 0 ≤ row) (hcolumn : 0 ≤ column) :
    2 - row - column ≤ 2 := by
  linarith

theorem reciprocal_complement_bound {a h : ℝ}
    (hh : 1 - a ≤ h) (hpos : 0 < h) :
    1 / h ≤ 1 + a / h := by
  apply (div_le_iff₀ hpos).2
  have hone : 1 ≤ h + a := by linarith
  calc
    1 ≤ h + a := hone
    _ = (1 + a / h) * h := by field_simp

theorem weighted_second_moment_bound
    {ι : Type*} [Fintype ι] (a w : ι → ℝ)
    (ha : ∀ i, 0 ≤ a i) (hw : ∀ i, 0 ≤ w i)
    (hpoint : ∀ i, w i ≤ 1 + a i * w i) :
    (∑ i, a i * w i ^ 2) ≤
      (∑ i, a i * w i) + (∑ i, a i * w i) ^ 2 := by
  have hterm : ∀ i, a i * w i ^ 2 ≤
      a i * w i + (a i * w i) ^ 2 := by
    intro i
    have hmul := mul_le_mul_of_nonneg_left (hpoint i)
      (mul_nonneg (ha i) (hw i))
    nlinarith
  have hsum :
      (∑ i, a i * w i ^ 2) ≤
        ∑ i, (a i * w i + (a i * w i) ^ 2) :=
    Finset.sum_le_sum fun i _ ↦ hterm i
  have hsquares :
      (∑ i, (a i * w i) ^ 2) ≤ (∑ i, a i * w i) ^ 2 := by
    simpa using
      (Finset.sum_sq_le_sq_sum_of_nonneg
        (s := Finset.univ) (f := fun i ↦ a i * w i)
        (fun i _ ↦ mul_nonneg (ha i) (hw i)))
  rw [Finset.sum_add_distrib] at hsum
  linarith

theorem alignment_closing_inequality {A B : ℝ}
    (hA : 1 / 2 ≤ A) (hB : B ≤ A + A ^ 2) :
    B ≤ 3 * A ^ 2 := by
  have hAnonneg : 0 ≤ A := by linarith
  have htwice : A ≤ 2 * A ^ 2 := by
    have hfactor : 0 ≤ A * (2 * A - 1) :=
      mul_nonneg hAnonneg (by linarith)
    nlinarith
  linarith

theorem squared_cosine_floor {A B : ℝ}
    (hBpos : 0 < B) (hA : 1 / 2 ≤ A) (hAB : B ≤ A + A ^ 2) :
    1 / 3 ≤ A ^ 2 / B := by
  have hclose : B ≤ 3 * A ^ 2 := alignment_closing_inequality hA hAB
  apply (le_div_iff₀ hBpos).2
  nlinarith

theorem raw_field_scale {α g e : ℝ}
    (hα : α ≠ 0) (he : e ≠ 0) :
    (α * g) / (α ^ 2 * e) = (1 / α) * (g / e) := by
  field_simp
  ring

theorem raw_field_sign_alignment {g e : ℝ}
    (he : 0 < e) :
    0 ≤ g * (g / e) := by
  have hid : g * (g / e) = g ^ 2 / e := by ring
  rw [hid]
  exact div_nonneg (sq_nonneg g) (le_of_lt he)

/-- CauchyLift v0.3 exact degree-0 scale invariance. -/
theorem cauchylift_v3_scale_invariance {α g d : ℝ}
    (hα : α ≠ 0) (hd : d ≠ 0) :
    (α * g) / (α * d) = g / d := by
  field_simp
  ring

/-- CauchyLift v0.3 strict positivity of fiber RMS denominator. -/
theorem cauchylift_v3_denom_pos {u v : ℝ}
    (hu : 0 < u) (hv : 0 ≤ v) :
    0 < u + v := by
  linarith

/-- CauchyLift v0.3 strict descent alignment on active coordinates. -/
theorem cauchylift_v3_strict_descent {g d : ℝ}
    (hg : g ≠ 0) (hd : 0 < d) :
    0 < g * (g / d) := by
  have hid : g * (g / d) = g ^ 2 / d := by ring
  rw [hid]
  exact div_pos (sq_pos_of_ne_zero hg) hd

/-- CauchyLift v0.3 coordinate magnitude bound: |g| / (u + v) ≤ √n when u² = r/n and |g| ≤ √r. -/
theorem cauchylift_v3_coordinate_bound {g u v n : ℝ}
    (hu : 0 < u) (hv : 0 ≤ v) (hg : |g| ≤ u * Real.sqrt n) :
    |g / (u + v)| ≤ Real.sqrt n := by
  have hden : 0 < u + v := by linarith
  have hpos : u ≤ u + v := by linarith
  rw [abs_div]
  have hden_abs : |u + v| = u + v := abs_of_pos hden
  rw [hden_abs]
  have hle : |g| / (u + v) ≤ |g| / u := by
    exact div_le_div_of_nonneg_left (abs_nonneg g) hu hpos
  have hstep : |g| / u ≤ Real.sqrt n := by
    exact (div_le_iff₀ hu).2 (by nlinarith)
  exact le_trans hle hstep

end CauchyLift


