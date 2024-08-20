## [4.1.1](https://github.com/act3-ace/safe-autonomy-sims/compare/v4.1.0...v4.1.1) (2024-08-20)


### Bug Fixes

* fixed negative inspected points reward bug ([#15](https://github.com/act3-ace/safe-autonomy-sims/issues/15)) ([4cf82bc](https://github.com/act3-ace/safe-autonomy-sims/commit/4cf82bc4f756b6da3fe5ec8f63d896b0d9a62102))

# [4.1.0](https://github.com/act3-ace/safe-autonomy-sims/compare/v4.0.1...v4.1.0) (2024-08-01)


### Features

* add gymnasium environments ([9664825](https://github.com/act3-ace/safe-autonomy-sims/commit/9664825c07e42a328d36dd58924a8b61f4e09ef8))

## [4.0.1](https://github.com/act3-ace/safe-autonomy-sims/compare/v4.0.0...v4.0.1) (2024-05-23)


### Bug Fixes

* six-dof field of view ([61403c7](https://github.com/act3-ace/safe-autonomy-sims/commit/61403c762cc2decf909083e2a4284d0573ecafd2))

# [4.0.0](https://github.com/act3-ace/safe-autonomy-sims/compare/v3.2.11...v4.0.0) (2024-04-25)


### Bug Fixes

* **dependencies:** update CoRL to 3.16.2, which absorbs some evaluation API ([1cfc057](https://github.com/act3-ace/safe-autonomy-sims/commit/1cfc057f3bfeb702a6d4b628550603c3d1a730a7))


### BREAKING CHANGES

* **dependencies:** Some evaluation API functions previously available from safe-autonomy-sims are now in the CoRL package and are no longer in the public API of safe-autonomy-sims

## [3.2.11](https://github.com/act3-ace/safe-autonomy-sims/compare/v3.2.10...v3.2.11) (2024-04-08)


### Bug Fixes

* **dependencies:** updated dynamics to 1.2.3 and rta to 1.16.1 ([65eb4da](https://github.com/act3-ace/safe-autonomy-sims/commit/65eb4da20d547cb97f3fc91a1f40b911a091518a))

## [3.2.10](https://github.com/act3-ace/safe-autonomy-sims/compare/v3.2.9...v3.2.10) (2024-03-27)


### Bug Fixes

* **dependencies:** Updated RTA to 1.16.0 ([70a0372](https://github.com/act3-ace/safe-autonomy-sims/commit/70a0372cad0d98ccc080e71ad0163911d8355905))

## [3.2.9](https://github.com/act3-ace/safe-autonomy-sims/compare/v3.2.8...v3.2.9) (2024-03-26)


### Bug Fixes

* **dependencies:** update CoRL to publicly released 3.14.13 ([ea1266e](https://github.com/act3-ace/safe-autonomy-sims/commit/ea1266ecf28135d13309b3ac7656baa324421553))

## 3.2.8 (2024-03-21)


### Bug Fixes

* **dependencies:** upgrade to CoRL 3.14.7

## 3.2.7 (2024-03-05)


### Bug Fixes

* inspection - rotate initial priority vector

## 3.2.6 (2024-02-01)


### Bug Fixes

* **dependencies:** update CoRL to 3.6.3 (public release candidate)

## 3.2.5 (2023-12-22)


### Bug Fixes

* Use rta singleton for rejection sampler

## 3.2.4 (2023-11-21)


### Bug Fixes

* Move horizon to env config

## 3.2.3 (2023-11-13)


### Bug Fixes

* Rotate priority vector based on entity

## 3.2.2 (2023-11-13)


### Bug Fixes

* Inspection-Inspector entity

## 3.2.1 (2023-11-07)


### Bug Fixes

* remove inspection chief assumption

# 3.2.0 (2023-11-07)


### Features

* Rejection Sampling

# 3.1.0 (2023-11-07)


### Features

* add angle to unit_vector transformation glue

# 3.0.0 (2023-11-07)


### Features

* Rejection Sampling
* update corl to 3.2.6


### BREAKING CHANGES

* The CoRL update introduces breaking changes, which propagates the breaking changes to the extensions provided by safe-autonomy-sims

## 2.13.1 (2023-10-09)


### Bug Fixes

* Count sensor single value 2b3b8a4

# 2.13.0 (2023-10-04)


### Features

* **6dof-inspection:** observations and rewards for the 6-dof inspection task 51d9f8a

## 2.12.2 (2023-09-08)


### Bug Fixes

* Make RTA module a singleton 6c5ee4d

## 2.12.1 (2023-09-08)


### Bug Fixes

* Make inspection delta-V scale a callback e6a30c4

# 2.12.0 (2023-08-10)


### Features

* Inspection Animation Updates 95d9eaf

# 2.11.0 (2023-07-22)


### Features

* Added num_workers argument to eval api functions to expose multiprocessing option. Found and fixed nondeterministic KMeans cluster function. 5aa395b

# 2.10.0 (2023-07-06)


### Features

* Resolve "evaluation api progress bar" 6853886

## 2.9.1 (2023-06-23)


### Bug Fixes

* inspection metric and animation e5614f8

# 2.9.0 (2023-06-22)


### Features

* inspection environment v2 (point weighting) 6d74ddd

# 2.8.0 (2023-06-21)


### Features

* make sun an entity 55bc1f1

## 2.7.1 (2023-06-19)


### Bug Fixes

* inspection animation and metric 8071e66

# 2.7.0 (2023-06-16)


### Features

* initializer can access attributes from the sim or other entities cb730be

# 2.6.0 (2023-06-16)


### Features

* support for multiagent inspection environments, velocity reference sensor, unit tests, and bug fixes. 8f0fd25

# 2.5.0 (2023-06-14)


### Features

* Update constraints for RTA e333ed3

## 2.4.1 (2023-06-08)


### Bug Fixes

* Resolve "eval api handle grid search configs" bb0ce8a

# 2.4.0 (2023-05-30)


### Features

* Add rl algs options to eval api 44a7592

## 2.3.5 (2023-05-17)


### Bug Fixes

* Resolve "measuring inspected points relies on the observation" 9624c68

## 2.3.4 (2023-05-09)


### Bug Fixes

* update to corl v2.9.0 5326350, closes #251

## 2.3.3 (2023-05-05)


### Bug Fixes

* update to corl v2.8.16 d2b8c6d, closes #249

## 2.3.2 (2023-04-24)


### Bug Fixes

* updated install.sh readme instructions b14a714

# 2.3.1 (2023-04-24)


### Bug Fixes

* Fixed pipeline for clean history 8ce8df9

# 2.3.0 (2023-04-20)


### Features

* 6DOF Inspection 297d059

# 2.2.0 (2023-04-19)


### Features

* Safety Metric db9fba6

# 2.1.0 (2023-04-17)


### Features

* Update corl version to 2.8.11 b470628

## 2.0.3 (2023-04-14)


### Bug Fixes

* pint conversions use default application registry. Initializer slow down fixed b3a7ddf

## 2.0.2 (2023-04-07)


### Bug Fixes

* removed repo links from semantic release commit msg 1a9fb77

## 2.0.1 (2023-04-06)


### Bug Fixes

* actually do name chnage

# 2.0.0 (2023-04-06)


### Features

* change package name to safe_autonomy_sims


### BREAKING CHANGES

* package name changed to safe_autonomy_sims

Merge branch '245-breaking-change-package-name-to-safe_autonomy_sims' into 'main'

Resolve "BREAKING: change package name to safe_autonomy_sims"

# 1.12.0 (2023-04-04)


### Features

* Support for rl algorithms

# 1.11.0 (2023-04-03)


### Features

* inspection environment updated with delta-v reward scheduler, unobserved points observation, and updated parameters

# 1.10.0 (2023-04-03)


### Features

* per entity initializers with unit handling via pint

## 1.9.2 (2023-03-30)


### Bug Fixes

* update seaborn to version 0.12

## 1.9.1 (2023-03-30)


### Bug Fixes

* reduce corl version restriction

# 1.9.0 (2023-03-21)


### Features

* remove dubins environments

# 1.8.0 (2023-03-16)


### Features

* added fast and performance training tests

## 1.7.1 (2023-03-15)


### Bug Fixes

* cwh3d docking performance fixed

# 1.7.0 (2023-03-15)


### Features

* CoRL update to 2.8.0 and Update agent actions based on frame rate

## 1.6.1 (2023-03-13)


### Bug Fixes

* added chief entity to inspection environments

# 1.6.0 (2023-03-10)


### Features

* Resolve "Make chief separate entity"

# 1.5.0 (2023-03-07)


### Features

* Resolve "Consolidate Thrust and Dubins controllers"

## 1.4.1 (2023-03-06)


### Bug Fixes

* Add Eval trial indices

# 1.4.0 (2023-02-28)


### Features

* add inspection illumination environment

## 1.3.3 (2023-02-28)


### Bug Fixes

* use agent frame rates to set sim frame rate

## 1.3.2 (2023-02-26)


### Bug Fixes

* fixed agent vs platform ID issue in RejoinSuccessReward, leveraged CoRL reference store in agent configs, resolved config variable discrepancies

## 1.3.1 (2023-02-24)


### Bug Fixes

* use frame_rate to update step_size

# 1.3.0 (2023-02-13)


### Features

* Inspection Animation script + obs/action Metrics + single checkpoint eval function

## 1.2.1 (2023-02-13)


### Bug Fixes

* Resolve "Evaluation Improvements"

# 1.2.0 (2023-02-09)


### Features

* Resolve "Moving to dev branch for CoRL"

## 1.1.1 (2023-01-26)


### Bug Fixes

* Simulator state sim time

# 1.1.0 (2023-01-24)


### Features

* updated eval API for corl 2.4.1

# 1.0.0 (2023-01-24)


* feat!: update corl to 2.4.1
* feat!: update corl to 2.4.1


### BREAKING CHANGES

* update corl 2.4.1
* update to corl 2.4.1

# 0.7.0 (2022-12-21)


### Bug Fixes

* **rta_rewards:** use next observation


### Features

* add inspection points sensor

## 0.6.1 (2022-12-12)


### Bug Fixes

* remove dt from illumination functions

# 0.6.0 (2022-12-01)


### Features

* add illumination to inspection environment

## 0.5.1 (2022-11-30)


### Bug Fixes

* trigger image build for ver 0.5

# 0.5.0 (2022-11-30)


### Bug Fixes

* fix access to parameter
* update to 1.58.0 [skip ci]
* update usage for dependent variables


### Features

* update corl
* update corl to 1.57.0 [skip ci]

# 0.4.0 (2022-11-29)


### Features

* inpsection problem and dependency updates

## 0.3.2 (2022-10-25)


### Bug Fixes

* configs and trianing test
* updat corl

## 0.3.1 (2022-10-18)


### Bug Fixes

* add tqdm
* ci resources
* remove glue refs
* remove unused docs
* update corl version
* update corl version
* update to corl normalizers

# 0.3.0 (2022-09-21)


### Bug Fixes

* update to use CoRL platform validator


### Features

* update CoRL to 1.51.0

# 0.2.0 (2022-08-30)


### Features

* **corl:** updated for corl 1.50.1

## 0.1.1 (2022-08-22)


### Bug Fixes

* a few minor fixes
* add markdownlint
* back to corl
* **Dockerfile:** create new package stages
* **Dockerfile:** fix pip install
* **Dockerfile:** update for package dependencies, remove git clone
* **Dockerfile:** update path
* **gitlab-ci:** add target stage to build job
* **gitlab-ci:** allow mkdoc to fail
* **gitlab-ci:** simplify jobs
* **gitlab-ci:** update job name
* **gitlab-ci:** update semantic release
* **gitlab-ci:** update to python unit test
* **gitlab-ci:** update to the very latest pipeline
* **gitlab-ci:** update unit_test_dir in job
* **gitlb-ci:** update pipeline version
* got back to python
* minor fix for corl
* permissions on opt dir
* **pipeline:** fix pylint
* **releaserc:** add for sematic release
* relocate version.py
* remove allowed to fail for mkdocs
* **setup.py:** update for package versioning
* **setup.py:** update path for version.py
* some reorganizing
* update jobs and docker file again
* update jobs and dockerfile
* update path on Dockerfile
* update tags
* update to latest cicd
* **update-version-python:** add file for versioning
* **version:** add version.py for semantic release
