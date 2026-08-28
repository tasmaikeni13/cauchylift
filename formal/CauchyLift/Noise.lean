import Mathlib

/-!
# Deterministic core of the stochastic alignment result

Expectation and measurability remain in the written probabilistic proof. This
file checks the pointwise scalar bookkeeping after Cauchy--Schwarz supplies the
sample/noise alignment terms.
-/

namespace CauchyLift

theorem noisy_alignment_lower
    {trueAlignment sampleAlignment errorAlignment γ ρ sampleNorm errorNorm : ℝ}
    (hdecomp : trueAlignment = sampleAlignment + errorAlignment)
    (hsample : γ * ρ * sampleNorm ≤ sampleAlignment)
    (herror : -(ρ * errorNorm) ≤ errorAlignment) :
    ρ * (γ * sampleNorm - errorNorm) ≤ trueAlignment := by
  linarith

theorem expected_alignment_margin_positive
    {ρ γ expectedSampleNorm expectedErrorNorm : ℝ}
    (hρ : 0 < ρ)
    (hmargin : expectedErrorNorm < γ * expectedSampleNorm) :
    0 < ρ * (γ * expectedSampleNorm - expectedErrorNorm) := by
  exact mul_pos hρ (sub_pos.mpr hmargin)

end CauchyLift
