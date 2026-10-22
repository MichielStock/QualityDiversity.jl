module QualityDiversityForwardDiffExt

using ForwardDiff: ForwardDiff
using QualityDiversity: QualityDiversity, AbstractQDProblem, evaluate

# The evaluation comes from the problem itself, so that it is exactly what `evaluate`
# returns, phenotype included: the primal part of the dual pass need not be bit-identical
# (reductions reassociate differently) and cannot carry a phenotype. The Jacobian is that
# of the fitness and descriptor components.
function QualityDiversity.forwarddiff_evaluate_gradient(prob::AbstractQDProblem, x::AbstractVector{<:Real})
    ev = evaluate(prob, x)
    jacobian = ForwardDiff.jacobian(y -> QualityDiversity.differentiable_values(prob, y), x)
    return ev, jacobian
end

end
