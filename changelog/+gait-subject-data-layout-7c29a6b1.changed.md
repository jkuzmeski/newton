Keep raw gait C3D inputs in each local `projects/gait_c3d/subjects/<subject>/`
bundle and preserve them when rebuilding a bundle with `--overwrite`. Document
the per-subject compilation and native motion-fitting workflow. Tracking
cluster centroids now average the remaining valid source markers when one
cluster label is absent or invalid, without fabricating a position. The native
motion example also reports IK preparation/JIT time separately from the
measured calculation rate.
