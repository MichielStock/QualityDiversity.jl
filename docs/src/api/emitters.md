```@meta
CurrentModule = QualityDiversity
```

# Emitters

An emitter proposes a batch of candidates each round and may learn from what the
archive did with them.

## Interface

```@docs
AbstractEmitter
QualityDiversity.ask
QualityDiversity.tell!
QualityDiversity.batch_size
QualityDiversity.restarts
```

## Variation emitter

```@docs
VariationEmitter
```

## CMA emitter

```@docs
CMAEmitter
```

### Evolution strategies

```@docs
AbstractEvolutionStrategy
CMAES
PyribsCMAES
SepCMAES
LMMAES
OpenAIES
QualityDiversity.reset!
QualityDiversity.sample!
QualityDiversity.update!
QualityDiversity.should_stop
QualityDiversity.RankOneRule
QualityDiversity.TextbookRankOne
QualityDiversity.PyribsRankOne
```

### Rankers

```@docs
AbstractRanker
QualityDiversity.rank_key
QualityDiversity.reset_ranker!
ObjectiveRanker
ImprovementRanker
TwoStageImprovementRanker
TwoStageObjectiveRanker
DirectionRanker
RandomDirectionRanker
TwoStageRandomDirectionRanker
AcceptedRandomDirectionRanker
```

### Selection rules

```@docs
QualityDiversity.SelectionRule
FilterSelection
MuSelection
```

### Restart rules

```@docs
QualityDiversity.RestartRule
NoImprovementRestart
ConvergenceRestart
PeriodicRestart
```

## Gradient emitters

Emitters for differentiable quality diversity, which step along the gradients of the
fitness and the descriptor; see [`evaluate_gradient`](@ref QualityDiversity.evaluate_gradient).
An emitter that needs the gradients at points of its own asks for them before every
round.

```@docs
GradientArborescenceEmitter
GradientOperatorEmitter
QualityDiversity.ask_gradients
QualityDiversity.tell_gradients!
QualityDiversity.GradientOptimizer
AdamAscent
GradientAscent
```

## Bandit scheduler

```@docs
BanditScheduler
QualityDiversity.active
QualityDiversity.ReselectRule
ReselectTerminated
ReselectAll
```
