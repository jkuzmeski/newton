# Compact marker-layout demonstration

These two small files have the shape of the offline outputs consumed by
`compile_subject_marker_layout()`:

- `adjusted_markers.xml` contains ten synthetic body-fixed marker placements.
- `body_transforms.json` contains matching source neutral body transforms.

They are a deterministic clean-checkout example fixture for the default simple
gait subject. They are not measured S001 data and are not an OpenSim validation
result. The full subject compiler produces the equivalent inputs with official
OpenSim offline, then publishes only neutral MJCF sites and a sealed marker
layout for Newton runtime use.
