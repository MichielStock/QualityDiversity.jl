@testset "GaussianMutation" begin
    op = GaussianMutation(0.5)
    parent = zeros(1000)
    child = QD.vary(Xoshiro(1), op, parent)
    @test child isa Vector{Float64}
    @test parent == zeros(1000)
    @test length(child) == 1000
    @test abs(sum(child) / 1000) < 0.1
    @test 0.4 < sqrt(sum(abs2, child) / 1000) < 0.6
    @test QD.vary(Xoshiro(1), op, parent) == child

    @test QD.vary(Xoshiro(1), GaussianMutation(1), [1, 2, 3]) isa Vector{Float64}
    @test QD.vary(Xoshiro(1), op, Float32[1, 2]) isa Vector{Float64}
    @test_throws "σ must be positive, got 0" GaussianMutation(0)
    @test_throws "σ must be positive, got -1.0" GaussianMutation(-1.0)
end

@testset "IsoLineVariation" begin
    op = IsoLineVariation(0.1, 0.5)
    @test QD.n_parents(op) == 2
    @test QD.n_parents(GaussianMutation(0.1)) == 1
    @test IsoLineVariation(0.1, 1) isa IsoLineVariation{Float64}
    @test IsoLineVariation() == IsoLineVariation(0.01, 0.2)
    @test IsoLineVariation(σ_line=0.5).σ_iso == 0.01

    p1 = [0.0, 0.0, 0.0]
    p2 = [1.0, 2.0, 3.0]
    child = QD.vary(Xoshiro(1), op, p1, p2)
    @test child isa Vector{Float64} && length(child) == 3
    @test p1 == [0.0, 0.0, 0.0] && p2 == [1.0, 2.0, 3.0]
    @test QD.vary(Xoshiro(1), op, p1, p2) == child
    @test !(child === p1) && !(child === p2)

    # no isotropic noise: the child is on the line through the parents
    line = IsoLineVariation(0.0, 1.0)
    for seed in 1:20
        c = QD.vary(Xoshiro(seed), line, p1, p2)
        @test c ≈ c[1] .* p2          # a multiple of p2 - p1, starting at p1
    end
    # no line term: Gaussian noise around the first parent
    iso = IsoLineVariation(0.3, 0.0)
    noise = reduce(vcat, [QD.vary(Xoshiro(s), iso, fill(2.0, 50), fill(9.0, 50)) .- 2.0 for s in 1:100])
    @test abs(sum(noise) / length(noise)) < 0.03
    @test 0.28 < sqrt(sum(abs2, noise) / length(noise)) < 0.32
    # both zero: a copy of the first parent
    @test QD.vary(Xoshiro(1), IsoLineVariation(0.0, 0.0), p1, p2) == p1
    # the line coefficient has standard deviation σ_line
    coefficients = [QD.vary(Xoshiro(s), IsoLineVariation(0.0, 0.5), [0.0], [1.0])[1] for s in 1:5000]
    @test abs(sum(coefficients) / 5000) < 0.03
    @test 0.47 < sqrt(sum(abs2, coefficients) / 5000) < 0.53

    @test QD.vary(Xoshiro(1), op, [1, 2, 3], Float32[4, 5, 6]) isa Vector{Float64}
    @test QD.vary(Xoshiro(1), IsoLineVariation(0.1f0, 0.2f0), Float32[1, 2], Float32[3, 4]) isa Vector{Float32}
    @test_throws "parents have 2 and 3 entries but must have the same number" QD.vary(Xoshiro(1), op, [1.0, 2.0], [1.0, 2.0, 3.0])
    @test_throws "σ_iso must be non-negative, got -0.1" IsoLineVariation(-0.1, 0.2)
    @test_throws "σ_line must be non-negative, got -1" IsoLineVariation(0.1, -1)
end
