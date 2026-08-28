# S001 base subject

This tracked neutral bundle is the canonical **S001** marker-placement reference for the Newton-native gait subject. It contains the 35-marker layout produced from the accepted offline OpenSim placement, the compiled Newton MJCF, and the 19 subject-specific non-colliding OBJ bone visuals.

The source placement and geometry were produced offline from the S001 gait2354 ScaleTool conversion. The source OpenSim/C3D files remain external; `model/manifest.json` records source and output hashes for the checked-in neutral assets. Runtime fitting does not import OpenSim.

The native subject scaler uses this bundle as its base. It applies one uniform length scale to body-local geometry, marker sites, body frames, contacts, and inertias, and a separate mass scale to inertial values.
