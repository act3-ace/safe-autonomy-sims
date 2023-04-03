# [1.10.0](https://github.com/act3-ace/safe-autonomy-sims/compare/v1.9.2...v1.10.0) (2023-04-03)


### Features

* per entity initializers with unit handling via pint ([cde0198](https://github.com/act3-ace/safe-autonomy-sims/commit/cde019811487f32cd3496af806473f701268acf6))

## [1.9.2](https://github.com/act3-ace/safe-autonomy-sims/compare/v1.9.1...v1.9.2) (2023-03-30)


### Bug Fixes

* update seaborn to version 0.12 ([f4e4417](https://github.com/act3-ace/safe-autonomy-sims/commit/f4e44173f37a5a22f72689591a3d8ed01503a198)), closes [#242](https://github.com/act3-ace/safe-autonomy-sims/issues/242)

## [1.9.1](https://github.com/act3-ace/safe-autonomy-sims/compare/v1.9.0...v1.9.1) (2023-03-30)


### Bug Fixes

* reduce corl version restriction ([61137e0](https://github.com/act3-ace/safe-autonomy-sims/commit/61137e091670913a51a0698148beab03934970e9)), closes [#241](https://github.com/act3-ace/safe-autonomy-sims/issues/241)

# [1.9.0](https://github.com/act3-ace/safe-autonomy-sims/compare/v1.8.0...v1.9.0) (2023-03-21)


### Features

* remove dubins environments ([9b4b56b](https://github.com/act3-ace/safe-autonomy-sims/commit/9b4b56b4277acf40eb33b94cc5db0ba72ba12d03))

# [1.8.0](https://github.com/act3-ace/safe-autonomy-sims/compare/v1.7.1...v1.8.0) (2023-03-16)


### Features

* added fast and performance training tests ([5512f47](https://github.com/act3-ace/safe-autonomy-sims/commit/5512f477be8fa4494c339a4f8f5177031dc347d5))

## [1.7.1](https://github.com/act3-ace/safe-autonomy-sims/compare/v1.7.0...v1.7.1) (2023-03-15)


### Bug Fixes

* cwh3d docking performance fixed ([c14e46d](https://github.com/act3-ace/safe-autonomy-sims/commit/c14e46de6c6e594269899d5d7a907691c2707353))

# [1.7.0](https://github.com/act3-ace/safe-autonomy-sims/compare/v1.6.1...v1.7.0) (2023-03-15)


### Features

* CoRL update to 2.8.0 and Update agent actions based on frame rate ([01663b2](https://github.com/act3-ace/safe-autonomy-sims/commit/01663b2eb9b885454e1b20088c72c39d65e9eaab))

## [1.6.1](https://github.com/act3-ace/safe-autonomy-sims/compare/v1.6.0...v1.6.1) (2023-03-13)


### Bug Fixes

* added chief entity to inspection environments ([d3637ac](https://github.com/act3-ace/safe-autonomy-sims/commit/d3637ac23375bf5932836417892ddfec74298def))

# [1.6.0](https://github.com/act3-ace/safe-autonomy-sims/compare/v1.5.0...v1.6.0) (2023-03-10)


### Features

* Resolve "Make chief separate entity" ([88d2806](https://github.com/act3-ace/safe-autonomy-sims/commit/88d28067f6af30e82f4b98d61d7482be47c3ffec))

# [1.5.0](https://github.com/act3-ace/safe-autonomy-sims/compare/v1.4.1...v1.5.0) (2023-03-07)


### Features

* Resolve "Consolidate Thrust and Dubins controllers" ([f4d761f](https://github.com/act3-ace/safe-autonomy-sims/commit/f4d761fa3511fda217ed36492fc36ca085422bd9))

## [1.4.1](https://github.com/act3-ace/safe-autonomy-sims/compare/v1.4.0...v1.4.1) (2023-03-06)


### Bug Fixes

* Add Eval trial indices ([6c0eb38](https://github.com/act3-ace/safe-autonomy-sims/commit/6c0eb386bdcb5ad2baa44fd0c4bfff8c8e6711a0))

# [1.4.0](https://github.com/act3-ace/safe-autonomy-sims/compare/v1.3.3...v1.4.0) (2023-02-28)


### Features

* add inspection illumination environment ([15999f3](https://github.com/act3-ace/safe-autonomy-sims/commit/15999f320bf6c065619a688e69ee318980fb93c4)), closes [#217](https://github.com/act3-ace/safe-autonomy-sims/issues/217)

## [1.3.3](https://github.com/act3-ace/safe-autonomy-sims/compare/v1.3.2...v1.3.3) (2023-02-28)


### Bug Fixes

* use agent frame rates to set sim frame rate ([241f159](https://github.com/act3-ace/safe-autonomy-sims/commit/241f1593ba15b78b9cf1fe2e5f1068bd909c4f29)), closes [#226](https://github.com/act3-ace/safe-autonomy-sims/issues/226)

## [1.3.2](https://github.com/act3-ace/safe-autonomy-sims/compare/v1.3.1...v1.3.2) (2023-02-26)


### Bug Fixes

* fixed agent vs platform ID issue in RejoinSuccessReward, leveraged CoRL reference store in agent configs, resolved config variable discrepancies ([ffe46df](https://github.com/act3-ace/safe-autonomy-sims/commit/ffe46df250df80599e06cd6ef6a8b300c9d72587))

## [1.3.1](https://github.com/act3-ace/safe-autonomy-sims/compare/v1.3.0...v1.3.1) (2023-02-24)


### Bug Fixes

* use frame_rate to update step_size ([95ce866](https://github.com/act3-ace/safe-autonomy-sims/commit/95ce8668c55f9591c1631fdb46524215feee6f4f)), closes [#224](https://github.com/act3-ace/safe-autonomy-sims/issues/224)

# [1.3.0](https://github.com/act3-ace/safe-autonomy-sims/compare/v1.2.1...v1.3.0) (2023-02-13)


### Features

* Inspection Animation script + obs/action Metrics + single checkpoint eval function ([a758762](https://github.com/act3-ace/safe-autonomy-sims/commit/a758762fb151db8bd34b5d7202b0d5769cd0144a))

## [1.2.1](https://github.com/act3-ace/safe-autonomy-sims/compare/v1.2.0...v1.2.1) (2023-02-13)


### Bug Fixes

* Resolve "Evaluation Improvements" ([aa99253](https://github.com/act3-ace/safe-autonomy-sims/commit/aa99253d04947776c15bdfbe74802a8450b14b28))

# [1.2.0](https://github.com/act3-ace/safe-autonomy-sims/compare/v1.1.1...v1.2.0) (2023-02-09)


### Features

* Resolve "Moving to dev branch for CoRL" ([231aca9](https://github.com/act3-ace/safe-autonomy-sims/commit/231aca9cd51a9bfe5099b5154b7ffc69a03cbc36))

## [1.1.1](https://github.com/act3-ace/safe-autonomy-sims/compare/v1.1.0...v1.1.1) (2023-01-26)


### Bug Fixes

* Simulator state sim time ([2bd954d](https://github.com/act3-ace/safe-autonomy-sims/commit/2bd954d74c9d1fcfce0a10e6b2061803c58f3cb6))

# [1.1.0](https://github.com/act3-ace/safe-autonomy-sims/compare/v1.0.0...v1.1.0) (2023-01-24)


### Features

* updated eval API for corl 2.4.1 ([4542d5f](https://github.com/act3-ace/safe-autonomy-sims/commit/4542d5f984b15d4513615c7d2e68f29924019021))

# [1.0.0](https://github.com/act3-ace/safe-autonomy-sims/compare/v0.7.0...v1.0.0) (2023-01-24)


* feat!: update corl to 2.4.1 ([09e4b2b](https://github.com/act3-ace/safe-autonomy-sims/commit/09e4b2b1cfbf18121625140b436d93cc74646d96))
* feat!: update corl to 2.4.1 ([83ac969](https://github.com/act3-ace/safe-autonomy-sims/commit/83ac96939d00ff5ed7421532e074ade2e3e0260e))


### BREAKING CHANGES

* update corl 2.4.1
* update to corl 2.4.1

# [0.7.0](https://github.com/act3-ace/safe-autonomy-sims/compare/v0.6.1...v0.7.0) (2022-12-21)


### Bug Fixes

* **rta_rewards:** use next observation ([a27dc11](https://github.com/act3-ace/safe-autonomy-sims/commit/a27dc113ffe39e28f23f26ee288d52f41faa8281))


### Features

* add inspection points sensor ([3947163](https://github.com/act3-ace/safe-autonomy-sims/commit/394716327335bc4c56179e52f49b4cdb34c4cd1e)), closes [#187](https://github.com/act3-ace/safe-autonomy-sims/issues/187)

## [0.6.1](https://github.com/act3-ace/safe-autonomy-sims/compare/v0.6.0...v0.6.1) (2022-12-12)


### Bug Fixes

* remove dt from illumination functions ([2fde8d0](https://github.com/act3-ace/safe-autonomy-sims/commit/2fde8d0cbfcb58729a236b81cc8f84f8508daefc))

# [0.6.0](https://github.com/act3-ace/safe-autonomy-sims/compare/v0.5.1...v0.6.0) (2022-12-01)


### Features

* add illumination to inspection environment ([3672eec](https://github.com/act3-ace/safe-autonomy-sims/commit/3672eecb58bc59f5ee5bc30cb897dae6531f9dab)), closes [#178](https://github.com/act3-ace/safe-autonomy-sims/issues/178)

## [0.5.1](https://github.com/act3-ace/safe-autonomy-sims/compare/v0.5.0...v0.5.1) (2022-11-30)


### Bug Fixes

* trigger image build for ver 0.5 ([7be1903](https://github.com/act3-ace/safe-autonomy-sims/commit/7be19035b6c77fb6ccc6042820563edac089405c))

# [0.5.0](https://github.com/act3-ace/safe-autonomy-sims/compare/v0.4.0...v0.5.0) (2022-11-30)


### Bug Fixes

* fix access to parameter ([98c53a9](https://github.com/act3-ace/safe-autonomy-sims/commit/98c53a9d07ff2374c2895a04d2b9712f77376af5))
* update to 1.58.0 [skip ci] ([d4b6474](https://github.com/act3-ace/safe-autonomy-sims/commit/d4b6474deb9e39e0e1454cc5f7dbdd80f86c0b17))
* update usage for dependent variables ([945b7c8](https://github.com/act3-ace/safe-autonomy-sims/commit/945b7c8d4c05663724bf1316fa56a1b46ba002ea))


### Features

* update corl ([b5dea79](https://github.com/act3-ace/safe-autonomy-sims/commit/b5dea79a642117d0b4f5f03090f1df5f343f48c1))
* update corl to 1.57.0 [skip ci] ([155b66e](https://github.com/act3-ace/safe-autonomy-sims/commit/155b66e39f9f8b050612c2331e2fca6fa6d9e446))

# [0.4.0](https://github.com/act3-ace/safe-autonomy-sims/compare/v0.3.2...v0.4.0) (2022-11-29)


### Features

* inpsection problem and dependency updates ([28b7cc1](https://github.com/act3-ace/safe-autonomy-sims/commit/28b7cc145a97b60c019481797f0bfa46fd0c49fe))

## [0.3.2](https://github.com/act3-ace/safe-autonomy-sims/compare/v0.3.1...v0.3.2) (2022-10-25)


### Bug Fixes

* configs and trianing test ([1295a94](https://github.com/act3-ace/safe-autonomy-sims/commit/1295a944931c2258cde905839a134762b40e66e3))
* updat corl ([0b16755](https://github.com/act3-ace/safe-autonomy-sims/commit/0b167555f0887ae708888b47ff9b9dbf7412fb0d))

## [0.3.1](https://github.com/act3-ace/safe-autonomy-sims/compare/v0.3.0...v0.3.1) (2022-10-18)


### Bug Fixes

* add tqdm ([b4d19e0](https://github.com/act3-ace/safe-autonomy-sims/commit/b4d19e0a6bbed72af73eaa341a92c2560c715d97))
* ci resources ([55af9bd](https://github.com/act3-ace/safe-autonomy-sims/commit/55af9bd025e22eac20cd44f7781a396e4a5271ac))
* remove glue refs ([2185eb2](https://github.com/act3-ace/safe-autonomy-sims/commit/2185eb2643577dd6d769d7419d574cc5029e7483))
* remove unused docs ([2155464](https://github.com/act3-ace/safe-autonomy-sims/commit/21554641b5638d7534cf4c68e68aa95a5160287a))
* update corl version ([d551706](https://github.com/act3-ace/safe-autonomy-sims/commit/d551706f92c141df827194161397639af72a976b))
* update corl version ([ac3aa5c](https://github.com/act3-ace/safe-autonomy-sims/commit/ac3aa5c667a079c74b3fa71c80cba0e918b197c9))
* update to corl normalizers ([b6ea43c](https://github.com/act3-ace/safe-autonomy-sims/commit/b6ea43cbaf02f45b6032d615d06bd24dabd5a2e0))

# [0.3.0](https://github.com/act3-ace/safe-autonomy-sims/compare/v0.2.0...v0.3.0) (2022-09-21)


### Bug Fixes

* update to use CoRL platform validator ([24b0cc4](https://github.com/act3-ace/safe-autonomy-sims/commit/24b0cc4f4f5629956e95be540399de8f694f0cb6))


### Features

* update CoRL to 1.51.0 ([22db10e](https://github.com/act3-ace/safe-autonomy-sims/commit/22db10e7d37487f0c05cf65dec89c1e242394e4d))

# [0.2.0](https://github.com/act3-ace/safe-autonomy-sims/compare/v0.1.1...v0.2.0) (2022-08-30)


### Features

* **corl:** updated for corl 1.50.1 ([5534e42](https://github.com/act3-ace/safe-autonomy-sims/commit/5534e42076f7a9e7a2ebcbb7161590c5de0e072a))

## [0.1.1](https://github.com/act3-ace/safe-autonomy-sims/compare/v0.1.0...v0.1.1) (2022-08-22)


### Bug Fixes

* a few minor fixes ([d9acc40](https://github.com/act3-ace/safe-autonomy-sims/commit/d9acc4094d9db60126c710c4b981af7dd012cdd8))
* add markdownlint ([9f6d5f1](https://github.com/act3-ace/safe-autonomy-sims/commit/9f6d5f115893f51f89434abd5a6b25cd37cb6e09))
* back to corl ([d2ad3be](https://github.com/act3-ace/safe-autonomy-sims/commit/d2ad3be6517ffa0b5ef9a4cbd3f930f2efc0d122))
* **Dockerfile:** create new package stages ([612e9d9](https://github.com/act3-ace/safe-autonomy-sims/commit/612e9d911bc2efcc90b277ac17008a634dff6d4b))
* **Dockerfile:** fix pip install ([b3bd6cd](https://github.com/act3-ace/safe-autonomy-sims/commit/b3bd6cd53bfd1d84381c87cfbf00c69f6560a3a0))
* **Dockerfile:** update for package dependencies, remove git clone ([56dc901](https://github.com/act3-ace/safe-autonomy-sims/commit/56dc901e1ee4d896e16f28e11781fb0df0ee3273))
* **Dockerfile:** update path ([78d2f2b](https://github.com/act3-ace/safe-autonomy-sims/commit/78d2f2b50b49f1949d09962c548080587c62abd1))
* **gitlab-ci:** add target stage to build job ([7bcce0f](https://github.com/act3-ace/safe-autonomy-sims/commit/7bcce0ff097e853dee83b7ffaf508757a03d0510))
* **gitlab-ci:** allow mkdoc to fail ([65b359f](https://github.com/act3-ace/safe-autonomy-sims/commit/65b359f38bda47bc59974467c52b48a51c2a42b2))
* **gitlab-ci:** simplify jobs ([9ee3318](https://github.com/act3-ace/safe-autonomy-sims/commit/9ee331826735ac54e42b56effaf415d71f61dafc))
* **gitlab-ci:** update job name ([cda8577](https://github.com/act3-ace/safe-autonomy-sims/commit/cda857725b836b200d8c4ac0cbfd07f1a6e182b5))
* **gitlab-ci:** update semantic release ([b96664d](https://github.com/act3-ace/safe-autonomy-sims/commit/b96664d1f252728456807ece16ce1f058cfd1e43))
* **gitlab-ci:** update to python unit test ([28d26aa](https://github.com/act3-ace/safe-autonomy-sims/commit/28d26aa7ef9e6336ce8bb99f97f9c95db2082b45))
* **gitlab-ci:** update to the very latest pipeline ([e6183b9](https://github.com/act3-ace/safe-autonomy-sims/commit/e6183b92a164000534690fbc69bbf83eed771bca))
* **gitlab-ci:** update unit_test_dir in job ([589857b](https://github.com/act3-ace/safe-autonomy-sims/commit/589857b86f98007ed80297add7cb405e9c9c49c3))
* **gitlb-ci:** update pipeline version ([0e6d3ca](https://github.com/act3-ace/safe-autonomy-sims/commit/0e6d3cacbd2f13ab4e621c6bf74874ac187c5ef5))
* got back to python ([e6ed89f](https://github.com/act3-ace/safe-autonomy-sims/commit/e6ed89f43479a2ae91e4f2b4f4eef376c5103d1f))
* minor fix for corl ([4d078e5](https://github.com/act3-ace/safe-autonomy-sims/commit/4d078e59268b4186afbd9ab3270f6cc82d444a68))
* permissions on opt dir ([4a15103](https://github.com/act3-ace/safe-autonomy-sims/commit/4a15103c10eaa7c538e75d8342c4bdbe0ab77006))
* **pipeline:** fix pylint ([a994e80](https://github.com/act3-ace/safe-autonomy-sims/commit/a994e803ff113cf3018cac0c08b9703f1bd852bb))
* **releaserc:** add for sematic release ([6994299](https://github.com/act3-ace/safe-autonomy-sims/commit/69942995b96b4929e2b141274e1d41f3cac54d01))
* relocate version.py ([8181fa7](https://github.com/act3-ace/safe-autonomy-sims/commit/8181fa785e5090ac6a1f172be82e96f8351ae3a3))
* remove allowed to fail for mkdocs ([e356b42](https://github.com/act3-ace/safe-autonomy-sims/commit/e356b4229d5efc313c77d057a250302f18e2ecb8))
* **setup.py:** update for package versioning ([af02d85](https://github.com/act3-ace/safe-autonomy-sims/commit/af02d850fe20dd44893b1924cd654cc96a28ccea))
* **setup.py:** update path for version.py ([638fea8](https://github.com/act3-ace/safe-autonomy-sims/commit/638fea8a91ce7914789a17543eeb1d7f46d6cc96))
* some reorganizing ([5cba326](https://github.com/act3-ace/safe-autonomy-sims/commit/5cba326544456326bb17dfc28b10ed0bae4ad0c7))
* update jobs and docker file again ([d4d6d90](https://github.com/act3-ace/safe-autonomy-sims/commit/d4d6d903fb5cb6169f5e091d4fae9d3aaf6c0787))
* update jobs and dockerfile ([39732ef](https://github.com/act3-ace/safe-autonomy-sims/commit/39732ef9371a8d9c137b88723bfd536117e6ea3c))
* update path on Dockerfile ([6735451](https://github.com/act3-ace/safe-autonomy-sims/commit/6735451f143f1cbc1bf84e7fa606280affad64a4))
* update tags ([19e84f4](https://github.com/act3-ace/safe-autonomy-sims/commit/19e84f42e5e8d3f0a7f01e663270f57885a30030))
* update to latest cicd ([4161e0e](https://github.com/act3-ace/safe-autonomy-sims/commit/4161e0eaa8616981a6d935d1d82dd729063ef500))
* **update-version-python:** add file for versioning ([82bc2ab](https://github.com/act3-ace/safe-autonomy-sims/commit/82bc2ab32951b0faa09825fe402c2843d8b1fdfa))
* **version:** add version.py for semantic release ([056520b](https://github.com/act3-ace/safe-autonomy-sims/commit/056520bcddd3148becf6e443ffb437030b3628c0))
