from soccer_predict_lab.features import update_elo

def test_home_win_increases_home_elo():
    h,a = update_elo(1500,1500,"H")
    assert h > 1500
    assert a < 1500

def test_zero_sum():
    h,a = update_elo(1500,1500,"D")
    assert round(h+a,8)==3000
