# Use Cases for Chop Saw

The use cases for this digital twin can be categorized in three different ways:
- Describing: where measurements from the real entity are utilized by the DT. Communicating product dimensions or usage status is a good example of a relaying-type utilization.
- Predicting: where models are used to emulate the state evolution of the real entity.
- Prescribing: where the DT influences the state of the real entity.

The DT, as much as possible, is developed independently of any use case (from the perspective that state behavior is independent of usage; e.g. a model of a saw should behave like a saw no matter how it's being used). However, because the model must be made at some fidelity, it becomes necessary to describe the ways the DT will be used so that the minimum sufficient model fidelity can be determined.

# Possible Questions this DT Could Answer
1. What is the best way to use the saw to maximize longevity while perfoming all possible work?
1. What is the best way to use the saw to maximize cut effeciency?
1. How likely are the following failure modes:
    1. Blade chipping/cracking
    1. Crashing
1. What injury risks are present with the saw, and how can these be minimized?
