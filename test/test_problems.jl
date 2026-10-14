@testset "QDProblem" begin
    prob = QDProblem(x -> (-sum(abs2, x), (x[1], x[2]));
                     init=rng -> randn(rng, 2), bounds=((-3, 3), (-3, 3)))
    ev = QD.evaluate(prob, [1.0, 2.0])
    @test ev.fitness == -5.0
    @test ev.descriptor == SVector(1.0, 2.0)
    @test QD.descriptor_bounds(prob) == ((-3.0, 3.0), (-3.0, 3.0))
    @test length(QD.random_solution(Xoshiro(1), prob)) == 2
    @test QD.fitness_offset(prob) == 0

    bad = QDProblem(x -> (0.0, (1.0, 2.0, 3.0)); init=rng -> [0.0], bounds=((0, 1), (0, 1)))
    @test_throws "descriptor has 3 dimensions but the problem bounds have 2" QD.evaluate(bad, [0.0])
    @test_throws "each bound must be a (lo, hi) pair" QDProblem(identity; init=identity, bounds=((1, 2, 3),))

    evs = QD.evaluate_batch(prob, [[1.0, 2.0], [0.0, 0.0]])
    @test [e.fitness for e in evs] == [-5.0, 0.0]
end

@testset "Rastrigin" begin
    prob = Rastrigin(4)
    ev = QD.evaluate(prob, zeros(4))
    @test ev.fitness == 0.0
    @test ev.descriptor == SVector(0.0, 0.0)

    ev = QD.evaluate(prob, [1.0, 3.0, -2.0, 2.0])
    @test ev.descriptor == SVector(2.0, 0.0)
    @test ev.fitness ≈ -(40 + sum(x^2 - 10cos(2π * x) for x in [1.0, 3.0, -2.0, 2.0]))

    # solutions outside the box are clipped before scoring
    @test QD.evaluate(prob, fill(100.0, 4)) == QD.evaluate(prob, fill(5.12, 4))
    @test QD.evaluate(prob, fill(5.12, 4)).descriptor == SVector(5.12, 5.12)

    @test_throws "solution has 3 entries but Rastrigin has dimension 4" QD.evaluate(prob, zeros(3))
    @test_throws "at least 2 dimensions" Rastrigin(1)

    # the offset makes every fitness non-negative
    xs = range(-5.12, 5.12; length=100_001)
    @test all(10 + x^2 - 10cos(2π * x) <= 41 for x in xs)

    x = QD.random_solution(Xoshiro(1), prob)
    @test x isa Vector{Float64} && length(x) == 4 && all(abs.(x) .<= 5.12)
end

@testset "Sphere" begin
    prob = Sphere(4)
    shift = 0.4 * 5.12
    # the optimum has fitness 100 and a descriptor of the sums of the parameters
    ev = QD.evaluate(prob, fill(shift, 4))
    @test ev.fitness ≈ 100 && ev.descriptor ≈ SVector(2shift, 2shift)
    # the corner of the box farthest from the optimum has fitness 0
    @test QD.evaluate(prob, fill(-5.12, 4)).fitness ≈ 0 atol = 1e-12
    @test 0 < QD.evaluate(prob, zeros(4)).fitness < 100
    # outside the box the fitness is negative, and a parameter is mapped back by 5.12 / θ
    ev = QD.evaluate(prob, [10.24, 0.0, -10.24, 0.0])
    @test ev.fitness < 0 && ev.descriptor ≈ SVector(0.5, -0.5)
    # the descriptor of the largest sums reaches the bounds
    @test QD.evaluate(prob, fill(5.12, 4)).descriptor ≈ SVector(10.24, 10.24)
    @test QD.descriptor_bounds(prob) == ((-10.24, 10.24), (-10.24, 10.24))
    @test QD.descriptor_bounds(Sphere(20))[1] == (-51.2, 51.2)
    @test QD.evaluate(Sphere(5), fill(1.0, 5)).descriptor ≈ SVector(2.0, 3.0)      # halves of 2 and 3

    x = QD.random_solution(Xoshiro(1), prob)
    @test x isa Vector{Float64} && length(x) == 4 && all(abs.(x) .<= 5.12)
    @test QD.random_solution(Xoshiro(1), prob) == x
    @test QD.fitness_offset(prob) == 0
    @test_throws "solution has 3 entries but Sphere has dimension 4" QD.evaluate(prob, zeros(3))
    @test_throws "Sphere needs at least 2 dimensions" Sphere(1)
end

