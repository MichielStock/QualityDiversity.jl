```@meta
CurrentModule = QualityDiversity
```

# Algorithms and solving

An algorithm is an archive and a tuple of emitters. The constructors below assemble the
usual combinations; [`QDAlgorithm`](@ref) takes any other.

## Algorithms

```@docs
QDAlgorithm
MAPElites
CMAME
CMAMAE
CMAMEGA
CMAMAEGA
OMGMEGA
DDS
MOME
MELS
```

## Running

```@docs
solve
init
step!
solve!
QDState
QDResult
```
