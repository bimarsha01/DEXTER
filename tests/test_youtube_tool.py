from tools.youtube_tool import play_youtube


def test_play_youtube_empty_query():
    assert play_youtube("") == "You must provide a search term."


def test_play_youtube_mode_not_supported():
    # unsupported mode
    resp = play_youtube("baby justin bieber", mode="stream")
    assert "Only 'browser' mode is supported" in resp
