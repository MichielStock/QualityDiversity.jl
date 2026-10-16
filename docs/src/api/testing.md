```@meta
CurrentModule = QualityDiversity
```

# Testing

`QualityDiversity.Testing` checks that a new problem, archive, variation or emitter
meets its contract. It depends only on the `Test` standard library, so a package that
defines such a type can call it from its own tests; see
[Extending the package](../guide/extending.md).

```@docs
QualityDiversity.Testing
QualityDiversity.Testing.test_problem
QualityDiversity.Testing.test_archive
QualityDiversity.Testing.test_front_archive
QualityDiversity.Testing.test_variation
QualityDiversity.Testing.test_emitter
```
