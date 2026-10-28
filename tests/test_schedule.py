from wf.corruption.schedule import calibrate_p_arm, schedule


def test_schedule_deterministic():
    f1 = schedule("gameA", 7, "favorable", 0.5)
    f2 = schedule("gameA", 7, "favorable", 0.5)
    seq1 = [f1(t, False) for t in range(20)]
    seq2 = [f2(t, False) for t in range(20)]
    assert seq1 == seq2


def test_eligibility():
    fav = schedule("g", 1, "favorable", 1.0)
    unf = schedule("g", 1, "unfavorable", 1.0)
    assert fav(0, True) is False
    assert fav(0, False) is True
    assert unf(0, False) is False
    assert unf(0, True) is True


def test_calibrate():
    p = calibrate_p_arm(0.5, 3.0, 30.0)
    assert abs(p - 0.2) < 1e-9
