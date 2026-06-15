# Logger and Evaluator Design - Speaker Notes

## Slide 1: Logger and Evaluator Design
Introduce the goal: this is not just an app demo. It is the measurement infrastructure for a single-device fine-tuning leaderboard.

## Slide 2: Problem: Runs Are Not Enough
Explain why raw training results are not enough. Without context and validation, loss and RSS numbers cannot be trusted or compared.

## Slide 3: Overall Pipeline
Walk through the pipeline. Training produces events; logger records them; evaluator validates and computes metrics; leaderboard shows final rows.

## Slide 4: Logger: Record the Facts
Emphasize separation of concerns. Logger records facts only. It should not rank or judge methods.

## Slide 5: Logger Output: Run Bundle
Describe the run bundle as the evidence package. Locked configs make results comparable; hash chain makes tampering detectable.

## Slide 6: Evaluator: Validate and Score
Explain evaluator as a post-run validator and metric calculator. It turns recorded evidence into leaderboard-ready rows.

## Slide 7: Logger vs. Evaluator
Contrast the two modules. Logger answers what happened; evaluator answers whether it is valid and comparable.

## Slide 8: Example Run
Use Qwen + QNLI as the concrete example. One run on one phone becomes one validated leaderboard row.

## Slide 9: Takeaway
Close with the key message: trustworthy leaderboard results require both recording and validation, not only training execution. Also mention that the model layer is being extended beyond the current GPT-2, Gemma, and Qwen families, with Llama and more model families planned under the same logger/evaluator contract.
