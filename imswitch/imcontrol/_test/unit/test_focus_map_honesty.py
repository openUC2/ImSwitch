#!/usr/bin/env python3
"""WP-11 — the focus map must not claim more than its points support.

Two points do not define a plane, nor do three on a line: infinitely many
planes contain them and lstsq quietly returns the minimum-norm one. The fit
then reported "plane, MAE 0.000" — a confident green badge exactly when the
surface was least trustworthy, because the residuals were measured against the
very points that had been fitted.

Runs standalone: `.venv/bin/python <this file>`.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../.."))

from imswitch.imcontrol.model.focus_map import FocusMap, fit_quality_text  # noqa: E402


def _fit(points, method="spline", **kwargs):
    fm = FocusMap(group_id="t", method=method)
    for x, y, z in points:
        fm.add_point(x, y, z)
    return fm, fm.fit(**kwargs)


def test_two_points_do_not_make_a_plane():
    fm, stats = _fit([(0, 0, 10.0), (100, 0, 12.0)])
    assert stats.method == "constant"
    assert stats.fallback_used and "cannot define a tilted plane" in stats.fallback_reason
    # Constant Z everywhere, not a ramp extrapolated off the end of the line.
    assert fm.interpolate(0, 0) == fm.interpolate(5000, 5000) == 11.0


def test_three_collinear_points_do_not_make_a_plane():
    _, stats = _fit([(0, 0, 10.0), (50, 0, 11.0), (100, 0, 12.0)])
    assert stats.method == "constant"
    assert "one line" in stats.fallback_reason


def test_many_collinear_points_do_not_make_a_surface():
    _, stats = _fit([(x, 0.0, 10.0 + x / 100) for x in range(0, 101, 20)])
    assert stats.method == "constant"
    assert "one line" in stats.fallback_reason


def test_three_spread_points_make_a_plane_and_say_what_is_missing():
    fm, stats = _fit([(0, 0, 10.0), (100, 0, 12.0), (0, 100, 14.0)])
    assert stats.method == "plane"
    assert "needs 4" in stats.fallback_reason
    # Exact interpolating plane through the three points.
    assert abs(fm.interpolate(100, 100) - 16.0) < 1e-6


def test_no_error_number_is_reported_below_four_points():
    for points in ([(0, 0, 1.0)],
                   [(0, 0, 1.0), (1, 0, 2.0)],
                   [(0, 0, 1.0), (100, 0, 2.0), (0, 100, 3.0)]):
        _, stats = _fit(points)
        assert stats.mean_abs_error is None, stats
        assert stats.r_squared is None
        assert "not enough points to validate" in fit_quality_text(stats)


def test_error_is_cross_validated_so_an_interpolating_fit_cannot_claim_zero():
    rng = np.random.default_rng(0)
    points = [(x, y, 10 + 0.01 * x + 0.02 * y + rng.normal(0, 0.5))
              for x in range(0, 101, 25) for y in range(0, 101, 25)]

    _, honest = _fit(points)
    _, in_sample = _fit(points, cross_validate=False)

    assert honest.error_is_cross_validated
    assert not in_sample.error_is_cross_validated
    # The spline passes through every point it was given, so the in-sample
    # number is ~0 however wrong the surface is between the points.
    assert in_sample.mean_abs_error < 0.01
    assert honest.mean_abs_error > 0.1
    assert "cross-validated" in fit_quality_text(honest)


def test_constant_method_is_honoured_and_reports_a_real_error():
    _, stats = _fit([(0, 0, 10.0), (100, 0, 12.0), (0, 100, 14.0), (100, 100, 16.0)],
                    method="constant")
    assert stats.method == "constant"
    assert stats.mean_abs_error > 0


def test_z_offset_travels_with_the_fit():
    fm = FocusMap(group_id="t", method="spline", z_offset=5.0)
    for x, y, z in [(0, 0, 10.0), (100, 0, 12.0), (0, 100, 14.0)]:
        fm.add_point(x, y, z)
    stats = fm.fit()
    assert stats.z_offset == 5.0
    assert abs(fm.interpolate(0, 0) - 15.0) < 1e-6


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
    print("all passed")
