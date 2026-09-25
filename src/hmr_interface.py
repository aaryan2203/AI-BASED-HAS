"""
BAS-HAR: Human Mesh Recovery (HMR) Interface

This module provides an abstract interface for future Human Mesh Recovery
integration. HMR enables orientation-agnostic 3D human pose estimation,
which is critical for microgravity environments where the astronaut's
body may not be aligned with any fixed "floor" reference.

Current Status: STUB / RESEARCH INTERFACE
- The interface is defined but not functionally implemented
- 2D pose data is passed through as-is
- See models/hmr/README.md for integration roadmap

Future Integration:
- SMPL/SMPL-X body model
- HMR2.0 or 4DHumans for mesh recovery
- Payload/rack coordinate system transformation
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class BodyMesh:
    """Represents a 3D human body mesh (SMPL/SMPL-X format).

    Attributes:
        vertices: (N, 3) array of 3D vertex positions
        faces: (F, 3) array of triangle face indices
        joints_3d: (J, 3) array of 3D joint positions
        pose_params: SMPL pose parameters (body_pose + global_orient)
        shape_params: SMPL shape parameters (betas)
        camera_translation: (3,) camera translation vector
        confidence: Overall mesh recovery confidence
    """
    vertices: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    faces: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), dtype=int))
    joints_3d: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    pose_params: np.ndarray = field(default_factory=lambda: np.zeros(72))
    shape_params: np.ndarray = field(default_factory=lambda: np.zeros(10))
    camera_translation: np.ndarray = field(default_factory=lambda: np.zeros(3))
    confidence: float = 0.0


@dataclass
class RackCoordinates:
    """Astronaut pose relative to the experiment rack coordinate system.

    In microgravity, the astronaut may be oriented in any direction.
    This transforms the body pose into the rack's reference frame,
    making activity recognition orientation-independent.

    Attributes:
        joints_rack_frame: (J, 3) joint positions in rack coordinates
        body_orientation: (3, 3) rotation matrix of body relative to rack
        distance_to_rack: Distance from body center to rack origin
        facing_rack: Whether the astronaut is facing the rack
    """
    joints_rack_frame: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    body_orientation: np.ndarray = field(default_factory=lambda: np.eye(3))
    distance_to_rack: float = 0.0
    facing_rack: bool = False


class HMRProvider(ABC):
    """Abstract interface for Human Mesh Recovery providers.

    Implement this interface to integrate a specific HMR model
    (e.g., HMR2.0, 4DHumans, PyMAF, CLIFF) into the BAS-HAR system.
    """

    @abstractmethod
    def estimate_3d_pose(self, frame: np.ndarray) -> list[BodyMesh]:
        """Estimate 3D body meshes from a single frame.

        Args:
            frame: BGR image as numpy array (H, W, 3)

        Returns:
            List of BodyMesh objects for each detected person
        """
        ...

    @abstractmethod
    def get_body_mesh(self, frame: np.ndarray, bbox: tuple[int, int, int, int]) -> BodyMesh | None:
        """Recover body mesh for a specific person crop.

        Args:
            frame: Full BGR image
            bbox: Person bounding box (x1, y1, x2, y2)

        Returns:
            BodyMesh for the cropped person, or None if recovery fails
        """
        ...

    @abstractmethod
    def transform_to_rack_coords(
        self,
        body_mesh: BodyMesh,
        rack_markers: np.ndarray | None = None,
    ) -> RackCoordinates:
        """Transform body mesh to rack coordinate system.

        Args:
            body_mesh: 3D body mesh in camera coordinates
            rack_markers: Optional (N, 3) array of known rack marker positions

        Returns:
            RackCoordinates with pose in rack reference frame
        """
        ...


class StubHMRProvider(HMRProvider):
    """Stub HMR provider that passes through 2D pose data.

    This is a placeholder for the MVP. It does not perform actual
    mesh recovery — it returns empty/default structures.
    """

    def __init__(self):
        logger.info(
            "StubHMRProvider initialized. HMR is NOT active. "
            "See models/hmr/README.md for integration instructions."
        )

    def estimate_3d_pose(self, frame: np.ndarray) -> list[BodyMesh]:
        """Returns empty list — no actual 3D estimation."""
        logger.debug("StubHMRProvider.estimate_3d_pose called (no-op)")
        return []

    def get_body_mesh(self, frame: np.ndarray, bbox: tuple[int, int, int, int]) -> BodyMesh | None:
        """Returns None — no actual mesh recovery."""
        logger.debug("StubHMRProvider.get_body_mesh called (no-op)")
        return None

    def transform_to_rack_coords(
        self,
        body_mesh: BodyMesh,
        rack_markers: np.ndarray | None = None,
    ) -> RackCoordinates:
        """Returns default RackCoordinates — no actual transformation."""
        logger.debug("StubHMRProvider.transform_to_rack_coords called (no-op)")
        return RackCoordinates()


def get_hmr_provider(provider_name: str = "stub") -> HMRProvider:
    """Factory function to get an HMR provider by name.

    Args:
        provider_name: One of 'stub', 'hmr2', '4dhumans', 'pymaf'

    Returns:
        An HMRProvider instance
    """
    providers = {
        "stub": StubHMRProvider,
    }

    if provider_name not in providers:
        logger.warning(
            "Unknown HMR provider '%s', falling back to stub", provider_name
        )
        return StubHMRProvider()

    return providers[provider_name]()
