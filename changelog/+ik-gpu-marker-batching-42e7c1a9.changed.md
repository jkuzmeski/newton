Improve native marker inverse kinematics performance by batching marker
objectives and solving mocap frames in Warp batches. Target data, forward
kinematics, predictions, costs, and limit diagnostics remain device-resident
until the final result copy. No migration is required.
