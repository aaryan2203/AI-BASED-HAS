# Human Mesh Recovery (HMR) Integration Guide

## Overview

Human Mesh Recovery (HMR) enables full 3D body mesh estimation from a single 2D image. This is particularly important for microgravity environments like the BAS space station, where astronauts may be oriented in any direction relative to the camera.

## Why HMR for Space?

In standard terrestrial HAR systems, the assumption is that gravity defines "up" and the person's body is generally aligned with the floor. In microgravity:

- Astronauts can be upside down, sideways, or at any angle
- Standard 2D pose estimation may misinterpret activities
- The experiment rack provides the true reference frame, not the floor

HMR solves this by recovering a full 3D body mesh that can be transformed into the rack's coordinate system, making activity recognition **orientation-independent**.

## Architecture

```
Camera Frame
    ↓
Person Detection (YOLO)
    ↓
Person Crop
    ↓
HMR Model (e.g., HMR2.0)
    ↓
SMPL Body Mesh (vertices, joints, pose params)
    ↓
Rack Coordinate Transformation
    ↓
Orientation-Independent Features
    ↓
Activity Recognition
```

## Recommended Models

| Model | Paper | Pros | Cons |
|-------|-------|------|------|
| HMR 2.0 | Goel et al., 2023 | Strong accuracy, ViT backbone | GPU required |
| 4DHumans | Goel et al., 2023 | Multi-person, temporal | Heavy compute |
| PyMAF-X | Zhang et al., 2023 | Hands+face+body | Complex setup |
| CLIFF | Li et al., 2022 | Good with occlusion | Moderate accuracy |
| WHAM | Shin et al., 2024 | World-grounded motion | Video-based |

## SMPL Body Model

The SMPL model represents the human body with:
- **Shape parameters (β)**: 10 coefficients controlling body shape
- **Pose parameters (θ)**: 72 values (24 joints × 3 axis-angle rotations)
- **Mesh**: 6890 vertices, 13776 faces
- **Joints**: 24 body joints in 3D

## Integration Steps

### 1. Install Dependencies
```bash
pip install smplx pyrender trimesh
# For HMR2.0:
pip install git+https://github.com/shubham-goel/4D-Humans.git
```

### 2. Download SMPL Model
- Register at https://smpl.is.tue.mpg.de/
- Download SMPL model files
- Place in `models/hmr/smpl/`

### 3. Implement HMRProvider
```python
from src.hmr_interface import HMRProvider, BodyMesh

class HMR2Provider(HMRProvider):
    def __init__(self, model_path: str):
        # Load HMR2.0 model
        self.model = load_hmr2_model(model_path)

    def estimate_3d_pose(self, frame):
        # Run HMR2.0 inference
        results = self.model(frame)
        return [self._to_body_mesh(r) for r in results]

    def get_body_mesh(self, frame, bbox):
        # Crop and run inference
        crop = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        result = self.model(crop)
        return self._to_body_mesh(result)

    def transform_to_rack_coords(self, body_mesh, rack_markers=None):
        # Transform joints to rack coordinate system
        # Use ArUco markers or known rack geometry
        ...
```

### 4. Register Provider
Add to `src/hmr_interface.py`:
```python
providers = {
    "stub": StubHMRProvider,
    "hmr2": HMR2Provider,  # Add here
}
```

### 5. Update Config
```json
{
    "models": {
        "hmr": {
            "provider": "hmr2",
            "model_path": "models/hmr/hmr2_checkpoint.pth",
            "smpl_path": "models/hmr/smpl/"
        }
    }
}
```

## Rack Coordinate System

To make activity recognition orientation-independent:

1. **Define rack origin**: Use ArUco markers or known geometry on the rack
2. **Detect rack pose**: Estimate 6-DOF pose of the rack from camera
3. **Transform body mesh**: Convert body joint positions from camera frame to rack frame
4. **Extract features**: Use rack-relative joint positions for activity recognition

```
Rack Frame:
  X → along rack width
  Y → along rack height
  Z → perpendicular to rack face

Astronaut joints in rack frame:
  - Right hand at (0.3, 0.8, 0.2) → near container position
  - Left hand at (0.1, 0.5, 0.1) → near rack body
```

## Current Status

- ✅ Interface defined (`src/hmr_interface.py`)
- ✅ Stub provider for MVP
- ⬜ SMPL model download
- ⬜ HMR2.0 integration
- ⬜ Rack coordinate transformation
- ⬜ Orientation-independent feature extraction
- ⬜ Performance optimization for edge deployment

## Performance Considerations

| Component | GPU Time | CPU Time |
|-----------|----------|----------|
| Person Detection | ~5ms | ~50ms |
| HMR2.0 | ~30ms | ~500ms |
| SMPL Forward | ~2ms | ~20ms |
| Coord Transform | <1ms | <1ms |

For real-time operation at 15+ FPS, GPU is recommended for HMR.

## References

1. Kanazawa et al., "End-to-end Recovery of Human Shape and Pose", CVPR 2018
2. Goel et al., "Humans in 4D", CVPR 2023
3. Loper et al., "SMPL: A Skinned Multi-Person Linear Model", SIGGRAPH 2015
