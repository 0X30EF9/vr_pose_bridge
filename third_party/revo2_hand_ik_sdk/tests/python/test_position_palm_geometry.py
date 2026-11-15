"""Pure-math regression checks for the v0.10.28 position-palm strategy."""
from __future__ import annotations

import math


def _dot(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _sub(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return tuple(x - y for x, y in zip(a, b))


def _cross(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _norm(a: tuple[float, float, float]) -> float:
    return math.sqrt(_dot(a, a))


def _unit(a: tuple[float, float, float]) -> tuple[float, float, float]:
    n = _norm(a)
    assert n > 1e-12
    return tuple(x / n for x in a)


def _scale(a: tuple[float, float, float], s: float) -> tuple[float, float, float]:
    return tuple(s * x for x in a)


def _add(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return tuple(x + y for x, y in zip(a, b))


def _palm_basis(root: tuple[float, float, float],
                index: tuple[float, float, float],
                little: tuple[float, float, float]) -> tuple[tuple[float, float, float], ...]:
    i = _sub(index, root)
    l = _sub(little, root)
    x = _unit(_sub(i, l))
    forward = _scale(_add(i, l), 0.5)
    forward = _sub(forward, _scale(x, _dot(x, forward)))
    y = _unit(forward)
    z = _unit(_cross(x, y))
    y = _unit(_cross(z, x))
    return x, y, z


def _local(root: tuple[float, float, float],
           basis: tuple[tuple[float, float, float], ...],
           point: tuple[float, float, float]) -> tuple[float, float, float]:
    delta = _sub(point, root)
    return tuple(_dot(axis, delta) for axis in basis)


def _rotate_z(point: tuple[float, float, float], angle: float) -> tuple[float, float, float]:
    c, s = math.cos(angle), math.sin(angle)
    return (c * point[0] - s * point[1], s * point[0] + c * point[1], point[2])


def test_position_palm_coordinates_ignore_global_translation_and_rotation() -> None:
    root = (0.0, 0.0, 0.0)
    index = (0.03, 0.08, 0.0)
    little = (-0.03, 0.08, 0.0)
    middle = (0.02, 0.03, -0.025)
    far = (0.04, 0.05, -0.045)

    b0 = _palm_basis(root, index, little)
    m0 = _local(root, b0, middle)
    f0 = _local(root, b0, far)

    shift = (0.8, -0.3, 1.1)
    angle = 1.2
    transform = lambda p: _add(_rotate_z(p, angle), shift)
    root1, index1, little1 = map(transform, (root, index, little))
    middle1, far1 = map(transform, (middle, far))
    b1 = _palm_basis(root1, index1, little1)

    m1 = _local(root1, b1, middle1)
    f1 = _local(root1, b1, far1)
    for before, after in zip(m0 + f0, m1 + f1):
        assert math.isclose(before, after, abs_tol=1e-10)


def test_position_palm_root_sign_and_middle_hinge_are_separate_features() -> None:
    # Open proximal direction along +Y and palm normal +Z. Flexing toward -Z
    # yields a negative signed root rotation; left-side flex_sign=-1 turns it
    # into positive closure while right-side flex_sign=+1 applies to the
    # mirrored position-palm normal convention.
    open_ray = (0.0, 1.0, 0.0)
    axis = _unit(_cross(open_ray, (0.0, 0.0, 1.0)))
    flexed_ray = (0.0, math.cos(0.7), -math.sin(0.7))
    signed = math.atan2(_dot(axis, _cross(open_ray, flexed_ray)), _dot(open_ray, flexed_ray))
    assert signed < 0.0
    assert math.isclose(max(0.0, -signed), 0.7, abs_tol=1e-9)

    # The middle->far segment creates a separate observable PIP-like hinge.
    distal = (0.0, math.cos(0.9), -math.sin(0.9))
    hinge = math.atan2(_norm(_cross(flexed_ray, distal)), _dot(flexed_ray, distal))
    assert math.isclose(hinge, 0.2, abs_tol=1e-9)
