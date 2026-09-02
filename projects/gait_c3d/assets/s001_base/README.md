# S001 base subject

This tracked neutral bundle is the canonical **S001** marker-placement reference for the Newton-native gait subject. It contains 27 native marker sites produced from the accepted offline OpenSim placement. The four bilateral thigh/shank tracking clusters are each represented by one centroid. The raw offline placement remains available in `adjusted_markers.xml`. It also contains the compiled Newton MJCF and the 19 subject-specific non-colliding OBJ bone visuals.

The source placement and geometry were produced offline from the S001 gait2354 ScaleTool conversion. The source OpenSim/C3D files remain external; `model/manifest.json` records source and output hashes for the checked-in neutral assets. Runtime fitting does not import OpenSim. The static calibrated writer uses a distal-sacrum/superior-head torso frame with C7, T10, sternum, clavicle, and acromion markers and adds three bounded rotational torso axes.

The base-bundle scaler remains available for simple uniform rescaling. The
static-calibration writer uses this bundle as a mesh and mass template, then
applies per-side segment scales and calibrated body frames to the MJCF.

The bundle also includes `model/segment_calibration.json`, generated from the
S001 static calibration C3D over 0.5--1.0 s with a 6 mm marker radius. It
contains the raw bilateral PSIS slope, CODA/Bell--Brand hip centers, per-side
endpoint frames, medial/lateral widths, and the flat-foot calibration policy.
`VSAC` is retained as the derived PSIS midpoint for compatibility.

The calibration uses `LHLX` and `RHLX` as the hallux (big-toe) endpoints.
The personalized writer rebases the pelvis meshes to the calibrated CODA pelvis
origin before scaling.
