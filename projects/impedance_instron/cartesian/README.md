# Visual3D and runner inputs

The historical `cartesian/` package name is retained for existing imports and
serialized schema names. It contains current preparation and input code, not a
Cartesian controller:

- `visual3d.py` reads Visual3D exports and source clocks.
- `prepare_visual3d.py` makes a stance reference from a selected window.
- `prepare_dataset.py` selects and splits the F01 stance windows.
- `data.py` and `profile.py` validate the saved observation and inertial inputs.
- `shoe.py` attaches the shared Digital Shoe column bed to the planar foot.

The runner model and its identification are in `hogan/`.
