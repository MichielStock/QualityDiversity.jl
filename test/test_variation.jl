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
