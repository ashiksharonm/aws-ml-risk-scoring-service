from src.decisioning.reason_codes import adverse_action_reasons, base_feature


def test_base_feature_strips_prefix():
    assert base_feature("num__pay_1") == "pay_1"
    assert base_feature("pay_1") == "pay_1"


def test_only_positive_contributions_become_reasons():
    names = ["num__pay_1", "num__limit_bal", "num__pay_amt1"]
    out = adverse_action_reasons(names, [0.9, -0.5, 0.2])
    assert [r["code"] for r in out] == ["R01", "R05"]


def test_protected_attributes_are_never_reasons():
    names = ["num__sex", "num__age", "num__marriage", "num__education", "num__limit_bal"]
    out = adverse_action_reasons(names, [2.0, 2.0, 2.0, 2.0, 0.1])
    assert [r["code"] for r in out] == ["R03"]


def test_features_sharing_a_reason_are_merged_and_ranked():
    names = ["num__bill_amt1", "num__bill_amt2", "num__pay_1"]
    out = adverse_action_reasons(names, [0.3, 0.4, 0.6])
    assert out[0]["code"] == "R04"
    assert out[0]["contribution"] == 0.7
    assert out[0]["features"] == ["bill_amt1", "bill_amt2"]


def test_top_n_and_applicant_values():
    names = ["num__pay_1", "num__pay_2", "num__limit_bal", "num__bill_amt1"]
    out = adverse_action_reasons(names, [4, 3, 2, 1], top_n=2, applicant={"pay_1": 3.0, "pay_2": 1.0})
    assert len(out) == 2
    assert out[0]["applicant_values"] == {"pay_1": 3.0}


def test_no_risk_raising_features_gives_empty_list():
    assert adverse_action_reasons(["num__pay_1"], [-1.0]) == []
