# Biomechanics Asset Provenance

This inventory records non-code assets added by the biomechanics work. A checksum
identifies the exact bytes under review; it does **not** grant redistribution
rights.

## Redistribution status

The Digital Instron measurements and footwear geometry were supplied for this
project, but the repository does not yet contain written redistribution terms.
They are therefore approved only for this fork's internal project work and must
be excluded from an upstream Newton pull request until the owner records the
source, rights holder, and redistribution approval here.

The Gait2354 model embeds author/publication credits in the XML, but this branch
does not record an immutable upstream source or license for either the model or
motion. Treat both files as not cleared for upstream redistribution until those
fields are added. The generated human-shoe `.osim` file is a deterministic
derivative of that model and has the same restriction.

The left shoe-last STL is a mirrored derivative of the right source STL, as
documented in `DigitalInstron/README.md`.

The Gait2354 and human-shoe entries describe historical assets on
`jkuzmeski/digital-shoe-standalone`. They are not included in the standalone
showcase integration on `main`.

## Checksums

| Path | Bytes | SHA-256 |
| --- | ---: | --- |
| `DigitalInstron/04-07-2025_FR3_140ms_Rearfoot_100Cycle.steps.tracking.csv` | 797245 | `279ee5503ebead6ae5dd4323da2dc7f8f6020c417e381191c3de57a6f01c0d50` |
| `DigitalInstron/04-07-2025_FR3_185ms_Fullfoot_100Cycle.steps.tracking.csv` | 813557 | `eba087f82e467b35812f36f31adba590f95bc004b474aea276876ab29f927ed1` |
| `DigitalInstron/FoamSweep/10_13_2026_Dev3Elite_160-40-320ms_800-400-2800N_Sweep/10_13_2026_Dev3Elite_160-40-320ms_800-400-2800N_Sweep.Stop.csv` | 826357 | `ec72d2c99ad9264cf6f42e826880f4a0860571992c90a36bd06dac52ac80332a` |
| `DigitalInstron/FoamSweep/10_13_2026_Dev3Elite_160-40-320ms_800-400-2800N_Sweep/10_13_2026_Dev3Elite_160-40-320ms_800-400-2800N_Sweep.im_gpf` | 3004508 | `0be785d393c5fb03a818d70f93a2bd3a84f14764e6442711afba20e89d741783` |
| `DigitalInstron/FoamSweep/10_13_2026_Dev3Elite_160-40-320ms_800-400-2800N_Sweep/10_13_2026_Dev3Elite_160-40-320ms_800-400-2800N_Sweep.log` | 3726 | `3c35bfe9614549c903634752642152e8909603d2b6d7d768bb5defd9d773d2f9` |
| `DigitalInstron/FoamSweep/10_13_2026_Dev3Elite_160-40-320ms_800-400-2800N_Sweep/10_13_2026_Dev3Elite_160-40-320ms_800-400-2800N_Sweep.specimen.txt` | 21 | `879271077124cd1978ab45f0320aa4abb1a4c246f5af52f7088882d8426744ea` |
| `DigitalInstron/FoamSweep/10_13_2026_Dev3Elite_160-40-320ms_800-400-2800N_Sweep/10_13_2026_Dev3Elite_160-40-320ms_800-400-2800N_Sweep.steps.tracking.csv` | 33774390 | `43e6172ac4823ef839ad82db9fce66930af2aaa5bc25df59141f9d6fc37cbddd` |
| `DigitalInstron/FoamSweep/10_13_2026_Dev3Elite_160-40-320ms_800-400-2800N_Sweep/10_13_2026_Dev3Elite_160-40-320ms_800-400-2800N_Sweep.steps.trends.csv` | 234823 | `14c9bc1932935bc81504ceed13ff411e78d0b08382ab69dd11eb2a0317e629f3` |
| `DigitalInstron/FoamSweep/10_13_2026_FR3_160-40-320ms_800-400-2800N_Sweep/10_13_2026_FR3_160-40-320ms_800-400-2800N_Sweep.Stop.csv` | 828642 | `7cab6659cc569ecdef51eb789d499e819d926e36112486a9f7441ce04a9ae106` |
| `DigitalInstron/FoamSweep/10_13_2026_FR3_160-40-320ms_800-400-2800N_Sweep/10_13_2026_FR3_160-40-320ms_800-400-2800N_Sweep.im_gpf` | 3004508 | `0be785d393c5fb03a818d70f93a2bd3a84f14764e6442711afba20e89d741783` |
| `DigitalInstron/FoamSweep/10_13_2026_FR3_160-40-320ms_800-400-2800N_Sweep/10_13_2026_FR3_160-40-320ms_800-400-2800N_Sweep.log` | 3726 | `6a4818471cecd72a55739f10a6d28b75c370a77652bcb38e0af2224eeaeb26d1` |
| `DigitalInstron/FoamSweep/10_13_2026_FR3_160-40-320ms_800-400-2800N_Sweep/10_13_2026_FR3_160-40-320ms_800-400-2800N_Sweep.specimen.txt` | 21 | `879271077124cd1978ab45f0320aa4abb1a4c246f5af52f7088882d8426744ea` |
| `DigitalInstron/FoamSweep/10_13_2026_FR3_160-40-320ms_800-400-2800N_Sweep/10_13_2026_FR3_160-40-320ms_800-400-2800N_Sweep.steps.tracking.csv` | 33803785 | `6693877adfacab1fce458660a00f393a8ee423cc77e124468df41bcc75049d2f` |
| `DigitalInstron/FoamSweep/10_13_2026_FR3_160-40-320ms_800-400-2800N_Sweep/10_13_2026_FR3_160-40-320ms_800-400-2800N_Sweep.steps.trends.csv` | 234863 | `5973aea508973de1f1a334c5de90321c31487ce2dbb27770ea1177ca706aa354` |
| `DigitalInstron/FoamSweep/FoamSweep.ip_proj` | 1576 | `f882d8fe2d7aeb08433879745aa6420a4835d2cdfb79215891f6205bbb150890` |
| `DigitalInstron/FoamSweep/FoamSweep.log` | 31064 | `c75bded4fc9adab13e3dfc53633fc1c86f72447641462423c3f436793cfffde7` |
| `DigitalInstron/Instron Shoe Last Size 9 6drop merged attachment 1.STL` | 903784 | `0a61767ae86bc5eb80333d4163b0ca1b0952a82649230816163c117d24552d35` |
| `DigitalInstron/Instron Shoe Last Size 9 6drop merged attachment 1 left.STL` | 903784 | `39b832c0011ea05b2b3e7afc65ad099e98d7ed2285b60607806be88a4bade508` |
| `DigitalInstron/puma-deviate-nitro-elite-3/left/column_grid.npz` | 17947 | `66d4625b2d49b19c718970323e9df1adb7da49e648a6fef227dc244f63f01a0f` |
| `DigitalInstron/puma-deviate-nitro-elite-3/left/manifest_geometry_fragment.json` | 206 | `7d624b3e0060d19a980201ebd20143d2aed1f2455334ed65d66319b6dfbc24e5` |
| `DigitalInstron/puma-deviate-nitro-elite-3/left/preflight.json` | 1496 | `c4c95171f9b162cc52a2ad6369634f94544cb58ce2d8d0b522704c51b5bfa49c` |
| `DigitalInstron/puma-deviate-nitro-elite-3/plate.json` | 402 | `0a9902f68010099a9ef93ac5a788506d4ae6e3198db56beed36878312f764a6a` |
| `DigitalInstron/puma-deviate-nitro-elite-3/right/column_grid.npz` | 17947 | `e25196ff817d13a6c15860205aff21857565e1dc0e1f7ab58dcf1bc27655bd23` |
| `DigitalInstron/puma-deviate-nitro-elite-3/right/manifest_geometry_fragment.json` | 209 | `16b0f7e5353578a607779be3c85ba1017007aadf164987f0e628d3520458960d` |
| `DigitalInstron/puma-deviate-nitro-elite-3/right/preflight.json` | 1498 | `5369557acb586a42c8dab39ade37bb12628fc372eb2f2657747582e0b4a2d516` |
| `DigitalInstron/puma-deviate-nitro-elite-3/shoe.json` | 1445 | `fd850585d99fc904ef316fde618113dfc6759149e49270fa6ddc8755cfc70745` |
| `DigitalInstron/puma-deviate-nitro-elite-3/source.glb` | 110359616 | `e95298dd5f5bfdc7b5246904485d796fd4142932c877b1a5263f53c434e46dec` |
| `DigitalInstron/puma-fast-r-nitro-elite-3/left/column_grid.npz` | 17530 | `296c908f14c4b44cb2464058b8ee5fe8742e181bed0453b8e1f724b4c6f297a7` |
| `DigitalInstron/puma-fast-r-nitro-elite-3/left/manifest_geometry_fragment.json` | 206 | `7d624b3e0060d19a980201ebd20143d2aed1f2455334ed65d66319b6dfbc24e5` |
| `DigitalInstron/puma-fast-r-nitro-elite-3/left/preflight.json` | 1486 | `85ca4699d9a595f308f308e1387495b7e909cb54359fba6a31aacaa496d44477` |
| `DigitalInstron/puma-fast-r-nitro-elite-3/plate.json` | 399 | `f137edbb71bed14f13c22b75517dcbfb98c0bf05a448691321c17481dcdc000d` |
| `DigitalInstron/puma-fast-r-nitro-elite-3/right/column_grid.npz` | 17529 | `6904fa0ab22f3025e8a0dfe82b11801320d43a5a1b470a725bb40c5eb12ffdd3` |
| `DigitalInstron/puma-fast-r-nitro-elite-3/right/manifest_geometry_fragment.json` | 209 | `16b0f7e5353578a607779be3c85ba1017007aadf164987f0e628d3520458960d` |
| `DigitalInstron/puma-fast-r-nitro-elite-3/right/preflight.json` | 1488 | `04e96bbb1a6ee0ae9485b0d4a7d27dee3ddcd13531cd6138ca332045386ae780` |
| `DigitalInstron/puma-fast-r-nitro-elite-3/shoe.json` | 1435 | `5cb88ea436bd0cc2b54ad7ca36644ae2a7f5b30870929e199b5a01cdb73305d4` |
| `DigitalInstron/puma-fast-r-nitro-elite-3/source.glb` | 110647048 | `539b4683e2314fedde22e19ce2660615f07b55158f7cb289ee266c2ac2932095` |
| `DigitalInstron/puma-fast-r-nitro-elite-3-3d-internal-wt-LR.obj` | 537625 | `9347e6ad2bdeb4c7152cf5b7c50f784a5875d45df8656a23a567b9fb72c4753e` |
| `newton/examples/assets/gait2354_subject01.osim` | 318906 | `d22c504cc48103a560c725a6d7e909528eac8fac497c2514cbded71014882a8d` |
| `newton/examples/assets/gait2354_subject01_walk.mot` | 86534 | `752a125afddfa442c76fe74807b66b3d8e3836cac3a15cd409283766bcf402d4` |
| `docs/images/examples/example_impedance_instron.jpg` | 5946 | `033d6098989d3f4adc7814b58770fd77c3c9b1864f19cd219d7e493d572be4e6` |
| `docs/images/examples/example_impedance_stiffness.jpg` | 3677 | `abd8a0853391f8e081781cc80fefb44dbcf248f495e6ab24805432c4a53ae4b0` |
| `DigitalInstron/digital_shoe_showcase/dev3/digital_shoe.json` | 4538772 | `8f3b579bd23b9d8ff147900eb06d61d8f7f5f575bed404629a895116c55ec903` |
| `DigitalInstron/digital_shoe_showcase/dev3/digital_shoe_right.json` | 4536392 | `b93ccd1f84c90f2fddf7048edf3476e7ff3446da0a1ba8b1df2deba515114b11` |
| `DigitalInstron/digital_shoe_showcase/dev3/drop.gif` | 7160376 | `31634d8ca1b76636b7521151723b0fdb7d4290721a4886bdb39391b8173ec146` |
| `DigitalInstron/digital_shoe_showcase/dev3/instron.gif` | 827215 | `2a3482144ff2f33ffa330cf001b1cd5514fed18a3acc8a02d3492f4c419e373a` |
| `DigitalInstron/digital_shoe_showcase/dev3/rocker.gif` | 2129092 | `7809060127e931af566da999ab65463ce1ce6830541fb1e7dae91bec9ccceba9` |
| `DigitalInstron/digital_shoe_showcase/dev3/validation_report.html` | 14314443 | `fa664e49ca98a55d6cae1dff500712df6beecdce8aa2e4b28f2b24c0a946b5f9` |
| `DigitalInstron/digital_shoe_showcase/digital_shoe.json` | 4530311 | `2fc95daeb4536f3fd43892039aefcec2d3816aa42f775f3ba0d5a9fef81a0ff4` |
| `DigitalInstron/digital_shoe_showcase/digital_shoe_right.json` | 4529050 | `dfe252e5d9e0bf5f26e6f9ed975341ad3bfa64ca58d9ce7e0689ed7ad6f86673` |
| `DigitalInstron/digital_shoe_showcase/drop.gif` | 7122837 | `3a54d85d5f9398cc20f8b81a39ccf0ec4e28a80d8e594c13c34702c01546fdcc` |
| `DigitalInstron/digital_shoe_showcase/instron.gif` | 794612 | `260b775bd368cbaed0eb54680c4f898f6fb8cb1dfc672d7b7c12d514bff97f8f` |
| `DigitalInstron/digital_shoe_showcase/rocker.gif` | 1942403 | `2c4d48ea9bab69b5284e4cf1becd1c10b6c999d03586c607a27a9a61193686cb` |
| `DigitalInstron/digital_shoe_showcase/validation_report.html` | 13971909 | `0082ad02e1bf987e54fd82efe6eb3d9fa1e6c191df22d9297be561126dcc8dee` |
| `projects/impedance_instron/hogan/material_sensitivity/tornado_data.json` | 2343 | `f851c1c8e27290383c9ef2118c534c97f3fb38d792bb88872d83931d7c3db0fd` |
| `projects/impedance_instron/hogan/material_sensitivity/tornado_marathon065.png` | 76346 | `d3516b952ffdd49974dfdea06cf7db8b51c3a9093da917be0acd71d3cef4fe5b` |
| `projects/impedance_instron/hogan/profiles/f01_66_5kg.json` | 1641 | `ca0b719ad91ce16ebabd657079179c6698c37a69a6fec558848601db5bbc9daf` |
| `projects/impedance_instron/hogan/report_site/assets/dev3_loops.png` | 271803 | `c2f54cedd0d1507f49b07b4f35c0393987c8b3e49d78cd4a08814106ff7fea2f` |
| `projects/impedance_instron/hogan/report_site/assets/dev3_loss_trend.png` | 138070 | `20f7e9feea3fc9eb6e462864348102098c5a1ddb7fe6dc4483194fae7604240f` |
| `projects/impedance_instron/hogan/report_site/assets/drop.gif` | 7122837 | `3a54d85d5f9398cc20f8b81a39ccf0ec4e28a80d8e594c13c34702c01546fdcc` |
| `projects/impedance_instron/hogan/report_site/assets/instron.gif` | 794612 | `260b775bd368cbaed0eb54680c4f898f6fb8cb1dfc672d7b7c12d514bff97f8f` |
| `projects/impedance_instron/hogan/report_site/assets/rocker.gif` | 1942403 | `2c4d48ea9bab69b5284e4cf1becd1c10b6c999d03586c607a27a9a61193686cb` |
| `projects/impedance_instron/hogan/report_site/data.json` | 43428 | `620f4a8394e9f30c17fecf5c3dea44e2aa78a7e3d3bb12e837bd0629f7b0aa16` |
| `projects/impedance_instron/hogan/report_site/index.html` | 221817 | `9e27d0834744b597fba61a4420f667ca24a7d98cdf4eca0a8b9957b38c3a0b30` |
| `projects/impedance_instron/hogan/report_site/instron.json` | 59708 | `e1058a6592b3cd7ca2f33f89eb81720c24ef5cf4b708524a2640120c2f252ec0` |
| `projects/impedance_instron/hogan/report_site/motion.json` | 104765 | `c9339d68dd6c39cf2dbe83df5304d31a10827daa3a778485784420a94c353768` |
| `projects/impedance_instron/hogan/report_site/stance.gif` | 872786 | `0c1be7c57fcf19ef6ca0a40cf9c417eee98dfeeca983d722604c54171badd5f2` |

## Impedance Instron derivatives

`docs/images/examples/example_impedance_instron.jpg` is an internal engineering
render of the supplied Digital Shoe, a fixed mechanical ankle driven by a heel
marker triangle, and a virtual COM using the local S001 running acquisition. It is not a photograph of a participant. It remains a
restricted footwear/motion derivative, not an upstream-cleared example image.

`outputs/impedance_instron/` is ignored and contains the reduced running profile,
CSV traces, reports, and recordings. The sealed profile records source C3D and
adapter hashes, running-event evidence, and the lack of established redistribution
rights. The acquisition shoe is not the modeled Puma; the user confirmed its
role as representative example input. The in-memory fixture-side and known
winding corrections do not alter the original calibrated artifact or certify
anatomical side. Do not add that directory or the source acquisition to an
upstream PR.

`docs/images/examples/example_impedance_stiffness.jpg` is an internal engineering
render of the two-stiffness rig, simulated pelvis-centroid proxy, and the same
supplied Digital Shoe and optical reference. It is not a participant photograph.
Like the earlier example image, it is a restricted footwear/motion derivative;
this local integration does not grant upstream redistribution rights.

## Required before upstreaming

For every retained asset, add its source URL or repository and immutable
revision, author or rights holder, license, redistribution permission, units and
coordinate frame, and all local processing steps. For human-derived data, also
record the applicable de-identification and consent basis. Do not replace
missing facts with assumptions.
