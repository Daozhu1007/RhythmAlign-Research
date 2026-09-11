"""Part-15 six synthetic oracle cases (shared implementation with the pipeline)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import oracle_validation as OV


def test_case_y_equals_s():
    assert OV.case_y_equals_s()["pass"]


def test_case_s_zero():
    assert OV.case_s_zero()["pass"]


def test_case_separate_regions():
    assert OV.case_separate_regions()["pass"]


def test_case_destructive_interference():
    assert OV.case_destructive()["pass"]


def test_case_continuous_tone():
    assert OV.case_continuous_tone()["pass"]


def test_case_impulse():
    assert OV.case_impulse()["pass"]
