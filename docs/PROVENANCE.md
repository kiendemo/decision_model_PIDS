# Provenance

The numbers in `results/` were produced by research code that is not under version control (its
directory is untracked in the research repository), so the record of what was vendored is this
snapshot: the sha256, size and modification time of every original file the repository ports,
vendors or rebuilds its result files from, taken read-only by `scripts/dev/snapshot_sources.py`
before any porting began. Paths are relative to the root of the research tree. Data artifacts
(answer caches, score files, cards, precedent caches, ground truth, tokenizer files) are recorded
file by file, with their source sha256, in the release bundle's `MANIFEST.json`
(`scripts/make_release.py`). The short derivation scripts of the porting session are kept, with
their sha256, in the gitignored `runs/salvage/` (`scripts/dev/salvage.py`, `SALVAGE.json`).

Upstream PIDSMaker: commit 2289cd9 (graph construction and transformation); the research fork the
numbers come from matches upstream tag 2.1.1 for the files it did not change. Laya: release 0.3.5.

Snapshot taken 2026-10-30; 130 of 130 files present.

<!-- SNAPSHOT:BEGIN -->

| file | sha256 | bytes | mtime |
|---|---|---|---|
| `PIDSMaker/pidsmaker/reflex/kcs1/state.py` | 104e2e98c29f79bed33f9de7c02eff9cfb6c1e981a9e6ceca84906f538c50cb4 | 45602 | 2026-10-16 09:18:57 |
| `PIDSMaker/pidsmaker/reflex/kcs1/select.py` | 6ee0325d2f7cb6f447390081097e8894c2e3df655ad5114edc234bea06cfc04c | 4885 | 2026-10-15 21:09:55 |
| `PIDSMaker/pidsmaker/reflex/kcs1/questions.py` | 860f830e3d4690a3417592f61de7cda1ce80e0e951da43066a3031b8122e75f2 | 8497 | 2026-10-16 07:54:57 |
| `PIDSMaker/pidsmaker/reflex/kcs1/engine.py` | d04a1a627d6516cc07187219eb663aef294979ea60265ee66bbc20fd28ab78e9 | 31401 | 2026-10-16 09:22:24 |
| `PIDSMaker/pidsmaker/reflex/kcs1/sanitize.py` | 74ea74b0c297243f9c0b9126888177fb41b68862f64f03d2c6f5bfa47dd41e77 | 7118 | 2026-10-15 21:09:35 |
| `PIDSMaker/pidsmaker/reflex/kcs1/profile.py` | 886a0e0a24fade395207cff54772427a012b3f1abcf95d21a74f2d8fc3691856 | 10995 | 2026-10-16 09:16:44 |
| `PIDSMaker/pidsmaker/reflex/kcs1/score.py` | 3d670243d9901704eb171bf277e1687477c181e2673d54e3c9b71c92a7a322d7 | 8605 | 2026-10-16 07:55:11 |
| `PIDSMaker/pidsmaker/reflex/kcs1/null.py` | b611063446f7c0e395090cf4ea3143029f018468b253535c09505384dcb2e5ea | 5205 | 2026-10-16 09:19:54 |
| `PIDSMaker/pidsmaker/reflex/kcs1/checks.py` | fdb121ca5135a590fd8e30cb8925b99649e7ec802d302d2977e35e5223d80a00 | 25109 | 2026-10-16 09:19:54 |
| `PIDSMaker/pidsmaker/reflex/kcs1/run.py` | 2d67a69eaa7a09399199e758788d78e2b08994642a277f6d160d95831c01716e | 28383 | 2026-10-20 16:29:40 |
| `PIDSMaker/pidsmaker/reflex/kcs1/audit.py` | ea2e97f0ff57d4fba3eb6de8fbd3b5fd45148e43727455cb792abaefc43b65bd | 29322 | 2026-10-15 22:37:47 |
| `PIDSMaker/pidsmaker/reflex/kcs1/evaluate.py` | 84487c6c41c3512dc783b591d6504eaf41f6a491a708e404b5cebf42f3a80c23 | 120213 | 2026-10-20 16:29:02 |
| `PIDSMaker/pidsmaker/reflex/kcs1/paired.py` | 3571a2b9eda0bdd628b1cae4e4c4d010aed3f6a5d95305179a3a02dfc63b2f3d | 13102 | 2026-10-20 15:50:43 |
| `PIDSMaker/pidsmaker/reflex/kcs1/SPEC.md` | 8625fa5500f8f733bad083012e931db9d9907e0fd480c73d47e22be6a57ef561 | 24861 | 2026-10-15 21:03:19 |
| `PIDSMaker/pidsmaker/reflex/kcs1/SPEC_E2.md` | 211517dad9e8d2efd710c4141e6a29571f5feb6beae6d7271dafa4e6c7077921 | 11050 | 2026-10-16 09:45:33 |
| `PIDSMaker/pidsmaker/reflex/habitus/__init__.py` | aea54fe85dcb0fd67289224235b8b2e38acbbbdddafb3d21a2effe3b321a8ddc | 11271 | 2026-10-21 20:47:06 |
| `PIDSMaker/pidsmaker/reflex/habitus/paths.py` | 51cf16bb40cdc1fb290778a70dab8b26d1ac5a65d9250ec8836ddc9711a0d9b4 | 9978 | 2026-10-21 16:34:03 |
| `PIDSMaker/pidsmaker/reflex/habitus/guard.py` | c36fbce0ec2905cca154aa8016a2a7c1c45f4bc46f9acb7c80c82ea29322bf74 | 13969 | 2026-10-21 19:19:53 |
| `PIDSMaker/pidsmaker/reflex/habitus/types.py` | 3c0a8130158b3aaaa4ff958624366188b84c6a3111c9d05d8a7b7a592e75a05f | 42017 | 2026-10-21 20:36:54 |
| `PIDSMaker/pidsmaker/reflex/habitus/vocab.py` | 7509c7b22ff3b71be595c67ff530dabbc63febb5ffea9445e785f22ebfca8be0 | 24536 | 2026-10-21 17:23:26 |
| `PIDSMaker/pidsmaker/reflex/habitus/attest.py` | 7f33765dbdd959cc4e87b6394e7cb56a725cf4b995c00f83eebf55965fe72b2c | 11001 | 2026-10-21 17:23:13 |
| `PIDSMaker/pidsmaker/reflex/habitus/ifc.py` | f9f03d598a4db0cadcc3e273a3ba51bed3a808182d9e34a5df6e636677ee95d8 | 14364 | 2026-10-21 17:04:43 |
| `PIDSMaker/pidsmaker/reflex/habitus/profile2.py` | e558588b2e2d8ea959a2b27f00a195bee8cd48716d2d678445b6c88ea2f93b34 | 26805 | 2026-10-21 18:57:08 |
| `PIDSMaker/pidsmaker/reflex/habitus/views.py` | e8b007449520b26cf3da65df92c417bc9ca3cfffca343010525045197f8973ec | 22747 | 2026-10-21 18:57:17 |
| `PIDSMaker/pidsmaker/reflex/habitus/targets.py` | fc93494baa9be619346ec38324b1c2f73d7c5c3c607bb0e0375a813069c8d8ba | 23317 | 2026-10-21 17:20:17 |
| `PIDSMaker/pidsmaker/reflex/habitus/questions.py` | 73f7dad8a662007abc41f9961bf82b01f6910c8ddfe8084f03cdc16369b29c6e | 18166 | 2026-10-21 17:04:06 |
| `PIDSMaker/pidsmaker/reflex/habitus/head.py` | 15d70ce44d8daad90314954041445c3e05edee09b697d8c568920a4b01eb6794 | 3867 | 2026-10-21 16:39:06 |
| `PIDSMaker/pidsmaker/reflex/habitus/export.py` | e309a101123b2502bd13fc4758d1d014a1599fa670c74aa659b743ca9a7b5c27 | 48167 | 2026-10-21 20:48:59 |
| `PIDSMaker/pidsmaker/reflex/habitus/sample.py` | 9c2f122ca7d9d2ccb7952d1b841da1489f4d5976b1769529c8c165675f196d01 | 17434 | 2026-10-21 17:27:42 |
| `PIDSMaker/pidsmaker/reflex/habitus/surprise.py` | fe89db67bb9d9566890e52b8797f50fc5c84719d7580d3740d5abab7f4904db3 | 20010 | 2026-10-21 21:33:57 |
| `PIDSMaker/pidsmaker/reflex/habitus/null2.py` | 1fab45a04bf4080993eef1a561b7253afc0330835a950af8c218f08f3f2d5327 | 19364 | 2026-10-21 19:12:50 |
| `PIDSMaker/pidsmaker/reflex/habitus/score.py` | 2c120ea2220ce9fc96ba8b478b9dd3098cb020f5024584c11f2ef8684eface78 | 13431 | 2026-10-21 19:12:50 |
| `PIDSMaker/pidsmaker/reflex/habitus/retrofit.py` | 257c8ca44724745b3735861bfe414a899def9f310d0a74998a2ad3fcfbc5b1dd | 20771 | 2026-10-21 19:07:34 |
| `PIDSMaker/pidsmaker/reflex/habitus/decide.py` | 71441a2c3db92af693a28dcf8adf980d9669f813c7c33de6de76afa7cc82889a | 18274 | 2026-10-21 17:02:52 |
| `PIDSMaker/pidsmaker/reflex/habitus/engine.py` | 7c5800c5ddf3211285b98c63f78e2df58511f91d71efc42d0fd78ac27b4541ab | 22213 | 2026-10-21 21:34:55 |
| `PIDSMaker/pidsmaker/reflex/habitus/heads.py` | 059df4e2d09f5ea3083f1ffbb4ca83038553405e561b6c3d2d9f07541ac8eacd | 49088 | 2026-10-21 21:36:25 |
| `PIDSMaker/pidsmaker/reflex/habitus/encoder.py` | ab1527d2e402fd9f8441176833a6ecb9fa815b39d783c90791fa22dfcc2e2ebd | 45680 | 2026-10-21 21:36:19 |
| `PIDSMaker/pidsmaker/reflex/habitus/encoder_mq.py` | 0849de308ecb0db3b7b37ab10c2ce6f302426921e8ea00d47fd5c021875a4c4e | 57048 | 2026-10-21 21:34:42 |
| `PIDSMaker/pidsmaker/reflex/habitus/arms.py` | 39fd6a984f81e9b4a24ed0a8de33b27d79ec88260705846f58904162a0ea6acd | 66706 | 2026-10-21 21:39:35 |
| `PIDSMaker/pidsmaker/reflex/habitus/train.py` | e63acd12cd6170e58fd8c27d3a99344c682ca0651d63f5ee4436a3565915cf2d | 41830 | 2026-10-21 20:24:08 |
| `PIDSMaker/pidsmaker/reflex/habitus/train_enc.py` | 78d5503c11ad945584851c50e10e547a184e5a5430e4964a42407cb856ef3f9d | 41586 | 2026-10-21 21:47:57 |
| `PIDSMaker/pidsmaker/reflex/habitus/run.py` | 297a1c3167682084c1e4044b713d8e6ea9347328f042fbe3e5f82a119f7d2447 | 65082 | 2026-10-22 17:29:38 |
| `PIDSMaker/pidsmaker/reflex/habitus/checks.py` | 917e2e46056b8d96ad9d120a061f47e600b73c2e84220b58f4dd2794b8d2c54f | 94245 | 2026-10-21 21:40:57 |
| `PIDSMaker/pidsmaker/reflex/habitus/evaluate.py` | ad880303a5213be1590088ff6ad8c753c1f39371b5f78413688c8fc85c92679b | 85673 | 2026-10-23 14:46:54 |
| `PIDSMaker/pidsmaker/reflex/habitus/SPEC_HB.md` | 9c9104d59c513d879691c4c5bf9ceb0f34afe53d8dc3366f4516a32e833dd037 | 57512 | 2026-10-21 21:41:20 |
| `PIDSMaker/pidsmaker/reflex/common.py` | 05ef9482eba0d54d950b6cefbc45c146340dc9ae0607c90d18121f0ce33dd266 | 2482 | 2026-10-14 20:18:59 |
| `PIDSMaker/pidsmaker/reflex/identity.py` | fbcf6f73c654cdcfea32f61074ec6a58536f86572967ad152a1f7bf8faf79aa5 | 4696 | 2026-09-22 02:45:21 |
| `PIDSMaker/pidsmaker/reflex/cards.py` | f06b66887c8af68b948b2700a12afb3f3b972f2570acb539c8e32e154ab03439 | 25011 | 2026-10-14 20:08:09 |
| `PIDSMaker/pidsmaker/reflex/ledger.py` | 8535069aad90e46ec7485183a4f91db0103672c38acd11b702bb8f21045c2ce8 | 26838 | 2026-10-14 13:41:52 |
| `PIDSMaker/pidsmaker/reflex/kshot.py` | 78b6de6425037ebd1709197f69a84bb25e12cd9e61482c18e87670faf4cf3c5d | 14350 | 2026-10-14 20:08:24 |
| `PIDSMaker/pidsmaker/reflex/s1/student_dec.py` | 927366cc927471a9d9d1c60bfeb0a03f9b39e9374f4d84fc4f9386ccb3e77586 | 8120 | 2026-10-07 23:13:04 |
| `PIDSMaker/pidsmaker/reflex/anchor/host.py` | 4db56f9c559f848ecc005579cbed555e9e24d37688f08b5d91594e5282e4df33 | 10280 | 2026-10-14 20:24:52 |
| `PIDSMaker/pidsmaker/reflex/anchor/atoms_hist.py` | b1365f3b0784c410a7b5bc892d7f239d7f9e0e13416cf57a47c64af2f4b38f16 | 22000 | 2026-10-14 22:13:56 |
| `PIDSMaker/pidsmaker/reflex/anchor/qc.py` | 878daf0a6a9cdf21cf49ae11ad96bd25c375c76527a088366d861df03db00e20 | 51052 | 2026-10-14 23:05:20 |
| `PIDSMaker/pidsmaker/reflex/anchor/evalkit.py` | 74bd6ab5b63f2ae0fd7ccebfdef3bbbec5b5538ff5e32f974a684bca57ff9b68 | 15142 | 2026-10-14 22:17:22 |
| `PIDSMaker/pidsmaker/utils/utils.py` | ab1e878fa56fb543fa5e67ff3af6e2118c6d9b3385569f5195fb72302d7d1c07 | 24799 | 2026-08-03 19:47:38 |
| `PIDSMaker/pidsmaker/config/config.py` | 3c06eadcf8bf43737c804fe61d2c48ee79d4faa86bc6a3f04105be9158d694b7 | 44395 | 2026-09-16 06:12:50 |
| `PIDSMaker/pidsmaker/attacks/evaluate.py` | 5944ad5265cb05386275eae31d5a794ccdf92f5fc65f35c7985e1cc8c6a405de | 22107 | 2026-08-03 14:42:59 |
| `PIDSMaker/pidsmaker/defense/context.py` | 530ea0abc2442bb177d95e4962e3107a4cb2e0b625d65b3e8edc8157b3b0876d | 19983 | 2026-09-16 06:43:53 |
| `PIDSMaker/pidsmaker/defense/semantics.py` | f5fbe647dffb293153850367f79247e2d8a540d97b7dda18d483900210ec24fe | 13604 | 2026-09-16 06:22:27 |
| `PIDSMaker/pidsmaker/defense/profiles.py` | 9cd3bb967e204dd349c27ab3b86af22043e1e8399ed37a5438eb9ee480f73295 | 3035 | 2026-09-16 06:21:10 |
| `PIDSMaker/pidsmaker/defense/kb/build_kg.py` | 6be3acb787c4efd2f71ea145db649c9ac6c152578815dac3e79d5267aa1612eb | 7169 | 2026-09-16 03:13:51 |
| `PIDSMaker/pidsmaker/defense/framework.py` | 9e4816d5e4edd6e3f8abe7f4654d97f3443d48f1337f1abf592db29101824dca | 12236 | 2026-09-16 13:45:53 |
| `PIDSMaker/build_graphs_only.py` | 64cc961e70a5767e6d65419f6312d0b7d5e0ebf1313b646c8de0bcc642e9df83 | 1473 | 2026-08-26 14:34:48 |
| `PIDSMaker/LICENSE` | c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4 | 11357 | 2026-08-03 14:42:59 |
| `PIDSMaker_upstream/pidsmaker/config/pipeline.py` | 006d3ca1298b287553a1372951974cbc1b007c89f84b8fbdac067eb4daa3c2da | 34630 | 2026-09-23 04:27:02 |
| `PIDSMaker_upstream/pidsmaker/config/config.py` | 9bbbe31a93302178d4d0fa89393f3ee3362ee25303911d7a88b42a1bf2688f67 | 43959 | 2026-10-20 17:52:07 |
| `PIDSMaker_upstream/pidsmaker/utils/utils.py` | 87db0639de13f4b43fd653703d7293420c42fc1c57bda5b2fca7d38612ed5708 | 23789 | 2026-09-22 09:15:15 |
| `PIDSMaker_upstream/LICENSE` | c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4 | 11357 | 2026-09-22 09:15:15 |
| `third_party/laya/laya/common.py` | f231d42fcec84da203222fcaa89c083b22776e00341e66e118183d754e1dcabf | 11440 | 2026-09-22 01:29:02 |
| `third_party/laya/LICENSE` | a6cba85bc92e0cff7a450b1d873c0eaa2e9fc96bf472df0247a26bec77bf3ff9 | 10173 | 2026-09-22 01:29:02 |
| `third_party/laya/pyproject.toml` | 752774e03d1148ebf5848db51d3e417bf05e687f54ed473cefe3d5f4361df4b6 | 1328 | 2026-09-22 01:29:02 |
| `PIDSMaker/tests/test_kcs1.py` | 989308288f81f1a9828ea99dedfdfac719e470f698d11a7c0911364598d53376 | 109435 | 2026-10-20 16:29:02 |
| `PIDSMaker/tests/kcs1_e1_snapshot.py` | 0cd33831c13efdaa4a41f43ecd238b9f7b30979ec4b211e1a45a9529cec1923f | 13463 | 2026-10-16 09:49:12 |
| `PIDSMaker/tests/kcs1_e1_snapshot.json.gz` | 27deef5e87a4b4067b3af92a9e2c78f267b4a503e4958c04f68b31fcd12549ed | 78613 | 2026-10-16 09:13:17 |
| `PIDSMaker/tests/habitus_fakes.py` | e1366bd7cad2ce95865ade9ddde2f499621097b1b16ce407581aeab80da4ff7b | 12904 | 2026-10-21 16:54:29 |
| `PIDSMaker/tests/test_habitus_infra.py` | 41c64a58136752b4c8efee6ab00b2b05380c41ed5d8b35ab804601ca0e8989c9 | 27091 | 2026-10-21 16:53:37 |
| `PIDSMaker/tests/test_habitus_reference.py` | 4d6d2e3090c2a55eab5b6dd265e7f5556aa563ebd6fe087e9a0e9ca819d3ca68 | 39462 | 2026-10-21 17:21:14 |
| `PIDSMaker/tests/test_habitus_views.py` | edb49c56d868c23512a3fd8dbf445a9e7f10bc50827b9cff49fe4e4290333364 | 50358 | 2026-10-21 18:57:17 |
| `PIDSMaker/tests/test_habitus_export.py` | b69dc8840b064d60bfdca9238cbac249a8b3627197cc93a4ec2461022193e055 | 47852 | 2026-10-21 19:10:27 |
| `PIDSMaker/tests/test_habitus_engine.py` | beff9cb0200a7f5684a14a491b44194a8528eb4cee990d0cd917a231123b7cda | 42089 | 2026-10-21 17:25:38 |
| `PIDSMaker/tests/test_habitus_heads.py` | 4b753de34ad92480f6763a14485ae5cdbd54b1dde99b97e4f9357ed7bba12215 | 40116 | 2026-10-21 21:53:30 |
| `PIDSMaker/tests/test_habitus_encoder.py` | 03aaf8143fa4b10cc7e6bb0b690f5aa3d03af417b3fcadbbd915d857e931530d | 37981 | 2026-10-21 21:43:38 |
| `PIDSMaker/tests/test_habitus_encoder_mq.py` | 2d4fd2d9299f75ae42c0796c09c006dcbed620d981499b125aa31db9cd64c245 | 34735 | 2026-10-21 20:08:16 |
| `PIDSMaker/tests/test_habitus_surprise.py` | f3b1e3abf385e580bf15d857db441039482aa9da69d68f125f03481368c7081e | 56354 | 2026-10-21 21:43:16 |
| `PIDSMaker/tests/test_habitus_score.py` | 3965bd34ca136b603f6857bc2ddb83a4b0dc9f720e2a22a1ed8fe9074f5621ae | 40432 | 2026-10-21 19:12:50 |
| `PIDSMaker/tests/test_habitus_cli.py` | 7220a8962b6798ed7bcdcf6bacfd18a44bbbed7b530ff6f93458a34b3c622c0f | 41462 | 2026-10-21 19:16:10 |
| `PIDSMaker/tests/test_habitus_confusion.py` | 8c0b768904cfda20e34fb937c74d07f8938ca1f1314dbcb3cd50f0d1ed507453 | 11264 | 2026-10-22 17:32:52 |
| `PIDSMaker/tests/test_habitus_arms.py` | ed3e1775b727db2be949627feb76193d7cc921f74a5486260d23ea2b0321b983 | 61565 | 2026-10-21 21:54:47 |
| `research/results/reflex/habitus/zeroshot_ablation.py` | 34de17b92ef87f2048bde148609e3faaafadfce7140efd847a07ab724ce01b9c | 20402 | 2026-10-29 18:45:14 |
| `research/results/reflex/habitus/zeroshot_ablation.csv` | c748e3e4642a8370ba5881478f90a596ea769b6c3d3695dc602bb5688a017858 | 43288 | 2026-10-29 18:45:14 |
| `research/results/reflex/habitus/zeroshot_ablation.md` | adf3fb87d1ebe9b9c414472614979bf4e496a668dd76b751bdc36e0ea8e8471d | 10667 | 2026-10-29 18:45:14 |
| `research/results/reflex/habitus/confusion_reference.md` | 807845ee663c606f36c0b12fc3cdd61eba0949b0ac401856403292e0eddc4cbd | 41774 | 2026-10-22 17:38:56 |
| `research/results/reflex/habitus/selection_metrics_3folds.csv` | e04b43f15367c24c846c85d6a90b4d24f071c459782e35ab16e8b02f186155f4 | 1973 | 2026-10-29 17:12:40 |
| `research/results/reflex/habitus/run_facts_20261029.md` | ffb152c5c54dc1403755cf64881045b3b4c446f56e2adedd626aef31fc293cb2 | 1554 | 2026-10-29 17:12:40 |
| `research/results/reflex/habitus/arms_pilot_loho-optc_h501.md` | 2a4b89d653603472f4a9bc7e71ead51ce175e91f114b0ea9639199d29e00e610 | 5432 | 2026-10-22 03:27:23 |
| `research/results/reflex/habitus/arms_smoke.md` | a1001a64c9451a088ccf2cde17bdfc02e32e241dd7aeca786bf14584596d7019 | 6296 | 2026-10-21 23:48:03 |
| `research/results/reflex/habitus/decision_models/make_csv.py` | 4b9662d0b55d1ae72a93d55378ade71589424fca630de49d05eb70a4243964fd | 6376 | 2026-10-23 15:19:38 |
| `research/results/reflex/habitus/decision_models/decision_models_best_mcc.csv` | fb88ab05af64ee707cf9aa8e39ba0a22b1c1681c2c8a56a33767d067f9a4a43a | 32866 | 2026-10-23 15:19:38 |
| `research/results/reflex/habitus/decision_models/decision_models_mcc_matrix_k2.csv` | 50fc03cc22ea948b78ba3e362414383bc955e928743bd40207ebce5c3c4a069e | 902 | 2026-10-23 15:19:38 |
| `research/results/reflex/habitus/decision_models/decision_models_mcc_matrix_allc-pre.csv` | 1f4b8d3636f613628332e4d57ea448e46455d60501f3209b44303427a66cb9eb | 820 | 2026-10-23 15:19:38 |
| `research/results/reflex/habitus/decision_models/decision_models_speed_per_fold.csv` | aef3183747d4d60e3e5361ee0003a0b9170d3825f799c7b8dab73ee838639cf1 | 8203 | 2026-10-23 15:19:38 |
| `research/results/reflex/habitus/decision_models/decision_models_speed_mean.csv` | 3d58a92a4fdb64cebd1186370eef0b1c57f70801c093cf26714d408b4e904871 | 1076 | 2026-10-23 15:19:38 |
| `research/results/reflex/habitus/method_stats/routestats.json` | 3d450c1b6aabd74a8220399c8af3bb563d448ade3d8c88919119336e7614be6b | 5428 | 2026-10-29 19:03:47 |
| `research/results/reflex/habitus/method_stats/checka_dev.json` | fbfb35ee4c4de0532159e2ad03cb8e73799ab7a93b7fa3f5705237fae221e7f1 | 935 | 2026-10-29 19:03:47 |
| `research/results/reflex/habitus/method_stats/example_state_ATLASV2_EDR_2022-07-19.json` | 06740fd18838c747d5d333651dd9caa3307f59a342a0a5020aacf2bd05b2092f | 3010 | 2026-10-29 19:03:47 |
| `research/results/reflex/habitus/method_stats/tok_ATLASV2_EDR_all_2022-07-19.json` | cff9354dc89808fefc8e9bc1ae975759d779840de8ae5f3d1604a0c4c2cd5f25 | 1051 | 2026-10-29 19:03:47 |
| `research/results/reflex/habitus/method_stats/tok_CADETS_E3_2_2018-04-06.json` | 7599168fdbf250474fd3482b2dfbb72a39fb7aa557cd034d1db066476f2b78de | 1028 | 2026-10-29 19:03:47 |
| `research/results/reflex/habitus/method_stats/tok_THEIA_E3_all_2018-04-10.json` | f59d9f15fe40f38c165db9f08ed9b2b5dac1cd0d6592adf83d7bd1645a28b65b | 1062 | 2026-10-29 19:03:47 |
| `paper/habitus/derive/derive_zeroshot.py` | 0f74946a915d3cdaf1707fae7931ab5091baa468aea498ef24c38605189504cc | 9211 | 2026-10-29 19:10:26 |
| `paper/habitus/derive/editor_cpu.py` | 5b16cbc991c6002f308738f7485287d473735d82406925879ed603bfb2addf58 | 22753 | 2026-10-29 19:48:57 |
| `paper/habitus/derive/editor_facts.py` | 329abf015f18572c1596a1499066e477b45999c865a2f20b6cc5b0cf059c456a | 11371 | 2026-10-29 19:53:26 |
| `paper/habitus/derive/route_cost.json` | bc31cfa42ea41d2bd223bf7c9d4c650bd0f46114359b72a7f035538fba81da6f | 19396 | 2026-10-29 19:09:46 |
| `paper/habitus/derive/editor/editor_cpu.json` | 87f199bf82ab5ea109d4171a7f8aafbdaad20fa96b55e0a51cf9b8e2e7056188 | 125878 | 2026-10-29 19:51:21 |
| `paper/habitus/derive/editor/budget_CADETS_E3.json` | 69208e2c2e8dd547c1380c266189486d3c63a7801b38259698949c656130360e | 12221 | 2026-10-29 19:44:33 |
| `paper/habitus/derive/editor/budget_THEIA_E3.json` | ecbf85dc246094ce6c8f2e95bde6a18f837f4f31b81e157289ccbbfce50dc1e9 | 11478 | 2026-10-29 19:45:11 |
| `paper/habitus/derive/editor/budget_CLEARSCOPE_E3.json` | 06486b23ffbae84accf2723687aeb41d9e120afb686bba014d78fe3bffcd89d4 | 9012 | 2026-10-29 19:45:01 |
| `paper/habitus/derive/editor/budget_optc_h501.json` | 53299ca3b542eae9a709c3e4a5e467afee4c960d83cf6ad18052cd80bcbefafb | 9363 | 2026-10-29 19:45:17 |
| `paper/habitus/derive/editor/budget_ATLASV2_EDR.json` | 363c6f48178ad14e6c69331e355e25423e307be40c1b242a4eb1aa5a4c1fc57c | 17093 | 2026-10-29 19:45:02 |
| `paper/habitus/derive/editor/boot_CADETS_E3.json` | caac06d47adda8657a87b98741f9fcd34ad0f32df9a0ffe8d0c2b4e647390a84 | 413764 | 2026-10-29 19:50:01 |
| `paper/habitus/derive/editor/boot_THEIA_E3.json` | e7afa4dc6ddc4eda993f38ae6eb326778889d7cdb00957bc9109fb702ca12664 | 413831 | 2026-10-29 19:50:42 |
| `paper/habitus/derive/editor/boot_CLEARSCOPE_E3.json` | 72ea66adc809220cb61eef65deb1fde4c7f17c9d0f77598a2b0a88e9e6f7bef0 | 413857 | 2026-10-29 19:49:26 |
| `paper/habitus/derive/editor/boot_optc_h501.json` | 1e387956a43b77ad28c52036452b288ecb7444441d8564d02e929f3dc0ad6fda | 413963 | 2026-10-29 19:51:20 |
| `paper/habitus/derive/editor/boot_ATLASV2_EDR.json` | f814cc7b3920d31bb931dcbe992872d5f75723ec6653e5f98c4adebafd05e81a | 413875 | 2026-10-29 19:49:36 |
| `paper/habitus/derive/editor/state_CADETS_E3.json` | 32fd4a1b2809822d622c9657f62a596ba147819f824d5433cdd0f87e0a481307 | 5109 | 2026-10-29 19:49:44 |
| `paper/habitus/derive/editor/state_THEIA_E3.json` | 43583e02016d3f49fc01e7a321c18ab47cf513f4d7d68ea1cc0d4dcbb04ea1a7 | 4325 | 2026-10-29 19:49:45 |
| `paper/habitus/derive/editor/state_CLEARSCOPE_E3.json` | 202d1415abd7c5819c62ca9c53598133123efdb5bc1537a1c8a1d1b61a5bca65 | 4130 | 2026-10-29 19:49:17 |
| `paper/habitus/derive/editor/state_optc_h501.json` | 5d6d26820e80aa1ed72a61e048f7e5e54d3c6f4a05e2bd778c499d8bfb256526 | 6022 | 2026-10-29 19:50:17 |
| `paper/habitus/derive/editor/state_ATLASV2_EDR.json` | 463f2f8084a2013ecafbed96970e3a909ea427c8071a33bc5cc04dc4367cfde4 | 5866 | 2026-10-29 19:49:25 |
| `paper/habitus/experimental_setup.md` | fdfb51bd012ad300729cfcfd1089d10708105c2467c1177eef5a5f61e4f512ea | 34183 | 2026-10-29 20:01:12 |

<!-- SNAPSHOT:END -->

## Session derivation scripts (runs/salvage/)

Copied verbatim by `scripts/dev/salvage.py` (names relative to the session directory; not published). `python scripts/dev/salvage.py --check` re-verifies them.

| file | sha256 | bytes |
|---|---|---|
| `build_facts.py` | d257ccc6888ecdfbd226a9e168a714ad65821a5408edd49890b09a45cf23302d | 12803 |
| `cache_only_check.py` | bf566a9c6e8f752cc5403b846c605153dcdbf399c652c9e8e4b2f67b44d4944c | 3263 |
| `cache_only_check_e1.py` | c8dad2d3ec328f0459d6d79e62267a4765641a4f2e16e952ce1729161e254cc2 | 1791 |
| `edit/base.py` | a25e3d6380e8d3dd8d68826f40b48abc0f7d996d4bfd459ab3d333859e5c9b4b | 1533 |
| `edit/derive.py` | 5376b22ce1bce8706c94497910533220a7541b5e3c61a797faf2fd342cb48282 | 2982 |
| `edit/more.py` | a73569f466a41ea782266ec2ad609265030238a113dbfd00081b0efa43173460 | 1468 |
| `edit/rho.py` | a50598c752c5a6b479489df44b633db9563bdc36ab8e44897359b1726a0c7de4 | 1528 |
| `edit/rules.py` | adf08ac16ed976e4be6ca107564c8b0b7b715f9ef457f52e5645163c0b322f54 | 1679 |
| `env.sh` | 7d3d7897cbf79a4a106775fe641e73109d398f4f3cb43ac24f0f2eade237072d | 456 |
| `minimal/hb_data/__init__.py` | e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855 | 0 |
| `minimal/hb_data/cards.py` | af71fc643636bd562c3a6c321b63d4e8bc227655c7e5c7c7699af7f25b580fa3 | 19355 |
| `minimal/hb_data/graphs.py` | 4f4e6c764982975e001f8f4926f8f11aa455b9fab32b175f547985a29453a6e9 | 2596 |
| `minimal/hb_data/identity.py` | e46f8408ff38d9368cda998702c378759d60b39e6654cee109022aeb46e4c04f | 4685 |
| `minimal/hb_data/profiles.py` | 9cd3bb967e204dd349c27ab3b86af22043e1e8399ed37a5438eb9ee480f73295 | 3035 |
| `minimal/hb_data/resolver.py` | 7a7934f5e87179010f38470ed48c439695141482adc2fc3393e2b0affdf89080 | 11991 |
| `parse_sources.py` | 1c359a690e62d86f587ee11b673cc236a52a3f795cfa33e8e0072114f3af29b3 | 6469 |
| `verify_all.py` | 22fcf8d1b27766f7a6a33dc91f77644654d4ef60df4ceb5f1be0a055f41cbbcc | 1642 |
