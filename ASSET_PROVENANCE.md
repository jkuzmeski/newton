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

## Checksums

| Path | Bytes | SHA-256 |
| --- | ---: | --- |
| `DigitalInstron/04-07-2025_FR3_140ms_Rearfoot_100Cycle.steps.tracking.csv` | 797245 | `279ee5503ebead6ae5dd4323da2dc7f8f6020c417e381191c3de57a6f01c0d50` |
| `DigitalInstron/04-07-2025_FR3_185ms_Fullfoot_100Cycle.steps.tracking.csv` | 813557 | `eba087f82e467b35812f36f31adba590f95bc004b474aea276876ab29f927ed1` |
| `DigitalInstron/Instron Shoe Last Size 9 6drop merged attachment 1.STL` | 903784 | `0a61767ae86bc5eb80333d4163b0ca1b0952a82649230816163c117d24552d35` |
| `DigitalInstron/Instron Shoe Last Size 9 6drop merged attachment 1 left.STL` | 903784 | `39b832c0011ea05b2b3e7afc65ad099e98d7ed2285b60607806be88a4bade508` |
| `DigitalInstron/puma-fast-r-nitro-elite-3-3d-internal-wt-LR.obj` | 537625 | `9347e6ad2bdeb4c7152cf5b7c50f784a5875d45df8656a23a567b9fb72c4753e` |
| `newton/examples/assets/gait2354_subject01.osim` | 318906 | `d22c504cc48103a560c725a6d7e909528eac8fac497c2514cbded71014882a8d` |
| `newton/examples/assets/gait2354_subject01_walk.mot` | 86534 | `752a125afddfa442c76fe74807b66b3d8e3836cac3a15cd409283766bcf402d4` |

## Required before upstreaming

For every retained asset, add its source URL or repository and immutable
revision, author or rights holder, license, redistribution permission, units and
coordinate frame, and all local processing steps. For human-derived data, also
record the applicable de-identification and consent basis. Do not replace
missing facts with assumptions.
