# Project Documentation

#### Pull Request Process

1. Create a new branch for your feature or bug fix, starting from `dev`.
2. Commit your changes and push your branch to the repository.
3. Squash your commits into a single commit before creating a pull request.
4. Create a pull request from your branch to the `dev` branch.
5. Wait for the code review and address any comments or feedback.

#### Naming Conventions

##### Branches

use the following naming convention for branches:

<trigram>/<branch-name>

example: `vg/feature-branch`

##### Commits

use the following naming convention for commits:

<type>(scope): <description>

example: `feat(client): add new feature`

types can be:

- feat: a new feature
- evol: an improvement or enhancement
- fix: a bug fix
- hotfix: a critical bug fix
- docs: documentation changes
- refact: code refactoring
- test: adding or updating tests
- ci: changes to CI/CD configuration
- wip: work in progress (for temporary commits)

#### CI/CD

The project uses GitHub Actions for continuous integration. The CI/CD pipeline runs on every push to the repository and checks the code quality, runs the tests, and builds the project.

- Check the pipeline run status.

#### Branching strategy

- Use `main` for stable releases.
- Use `dev` for ongoing development.
- Create feature branches from `dev` for new features.
- Create hotfix branches from `master` for urgent fixes.