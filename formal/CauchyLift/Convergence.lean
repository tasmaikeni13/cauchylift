import Mathlib

/-!
# Descent-summation algebra

The analytic smoothness inequality is an assumption here.  This file verifies
the division and constant bookkeeping used to turn the summed decrease into a
stationarity bound.
-/

namespace CauchyLift

theorem generic_stationarity_bound
    {Δ L η ρ γ T M : ℝ}
    (hγ : 0 < γ) (hη : 0 < η) (hρ : 0 < ρ) (hT : 0 < T)
    (hsum : γ * η * ρ * T * M ≤ Δ + L * η ^ 2 * ρ ^ 2 * T / 2) :
    M ≤ Δ / (γ * η * ρ * T) + L * η * ρ / (2 * γ) := by
  have hden : 0 < γ * η * ρ * T := by positivity
  have hidentity :
      (Δ / (γ * η * ρ * T) + L * η * ρ / (2 * γ)) *
          (γ * η * ρ * T) =
        Δ + L * η ^ 2 * ρ ^ 2 * T / 2 := by
    field_simp
    ring
  apply (mul_le_mul_right hden).mp
  calc
    M * (γ * η * ρ * T) = γ * η * ρ * T * M := by ring
    _ ≤ Δ + L * η ^ 2 * ρ ^ 2 * T / 2 := hsum
    _ = (Δ / (γ * η * ρ * T) + L * η * ρ / (2 * γ)) *
          (γ * η * ρ * T) := hidentity.symm

end CauchyLift
