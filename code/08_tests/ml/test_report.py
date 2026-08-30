from ink_tafel_study.report import _property_panel_gate_is_eligible


def test_property_panel_report_gates_are_exact_and_fail_closed() -> None:
    assert _property_panel_gate_is_eligible(
        "eligible_complete_case_only", "eligible_complete_case_only"
    )
    assert _property_panel_gate_is_eligible(
        "eligible_sensitivity_only", "eligible_sensitivity_only"
    )

    assert not _property_panel_gate_is_eligible(None, "eligible_complete_case_only")
    assert not _property_panel_gate_is_eligible("", "eligible_complete_case_only")
    assert not _property_panel_gate_is_eligible(
        "eligible_sensitivity_only", "eligible_complete_case_only"
    )
