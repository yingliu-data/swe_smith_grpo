# ML Systems Engineer take-home

Your task is to design and build a small reinforcement learning (RL) training environment and a synthetic data generation pipeline that can be used to train a coding model on real-world software engineering (SWE) tasks.

The system you build must:
- Represent real SWE tasks in a structured, reproducible format
- Provide a stateful execution environment where an agent can interact with a codebase
- Support multi-step tool use (e.g. reading files, editing code, running tests)
- Generate or curate training data from real repositories
- Include an evaluation mechanism that could serve as a reward signal

For the purposes of this task you may assume that the training algorithm is GRPO, or one of its variants.

## Key components

You are expected to produce at minimum four key components:

1. Task and data pipeline
    - Design a pipeline that can be used to curate training tasks from real-world codebases. An example task schema is given here https://github.com/laude-institute/harbor-datasets/tree/main/datasets/swebench-verified/django__django-13346. Note that Harbor have constructed a single example as a directory, where instruction.md is the initial user prompt, etc. More information on this particular schema and the motivations for it can be found here https://harborframework.com/docs/tasks, however you do not need to exactly replicate this format, and may simplify or adapt it. Design and build an automated pipeline that can be used to create additional tasks given the schema you have been provided with. These tasks should be scoped specifically to bug fixes in the FastAPI repo, https://github.com/fastapi/fastapi. You are expected to generate at least 5 tasks using this automated pipeline. You can structure these tasks as directories, or you can collapse this structure into JSONL. Each task should include at least:
        - Repository reference
        - Base commit
        - Problem description/user prompt
        - Test commands
        - Reference solution (patch or commit)
2. RL environment and agent loop
    - A stateful environment in which the agent can attempt to solve a task. This should be a containerized solution, where the codebase is checked out at a specific commit, specific to the task at hand. In addition, you should integrate an agent loop that performs inference and allows the agent to interact with the environment. You can design your own, or use something from off-the-shelf like opencode. You are provided with a stub python interface as a minimal example, but you may use any interface or language you want to create this loop. You will be supplied with GPU credits that will allow you to host a model, using an inference engine of your choice, to test your agent loop and its integration with your RL environment. You are not being evaluated on the ability of the base model to solve the tasks, however the loop should be bug free.
3. Evaluators for rewards
    - Implement an evaluation mechanism that determines whether a task has been successfully solved. At minimum you should run the relevant tests that indicate whether the agent generated patch fixes the bug, and returns a reward signal.

You are encouraged to use a coding assistant to complete this challenge - we're interested in how you think, architect and build these systems. As such, please include a brief README.md that documents your design decisions, the trade-offs you made, areas for improvement, and how you could scale this solution to a large number of tasks across a training run.

## Guidance

We're evaluating system design, code quality and your grasp of ML concepts. The task is deliberately open-ended, and we encourage you to take it in any direction you want, but please spend no more than 5 hours total on it - a minimal, well-designed solution is preferred over a large, complex fragile one and, similarly, we don't mind if you integrate off-the-shelf tools instead of building your own components from scratch, so long as it is backed up by good systems design first-principles. Stubbed code provided as part of this task is meant only to provide ideas, and does not need to be used in preference to a 3rd party package or library to do well on this task.

You are expected to submit:
- Code in the form of a zipped file, or a link to a private GH repository.
- A README.md documenting your design choices.
- A dataset of the generated tasks, either in the directory or JSONL format mentioned above.