import math

from sentiment.preprocess import dedup_key, normalize_tweet


def test_normalize_handles_urls_entities_whitespace():
    s = normalize_tweet("@JoeBiden  wins &amp; more https://t.co/xyz  now")
    assert s == "@user wins & more http now"


def test_normalize_keeps_existing_user_token_and_emails_untouched():
    assert normalize_tweet("@user hi") == "@user hi"
    assert normalize_tweet("mail me a@b.com") == "mail me a@b.com"


def test_normalize_non_strings():
    assert normalize_tweet(None) == ""
    assert normalize_tweet(math.nan) == ""
    assert normalize_tweet(123) == "123"


def test_dedup_key_collapses_retweets_and_case():
    a = dedup_key("RT @someone: Trump WINS!! http://t.co/a")
    b = dedup_key("trump wins http://t.co/b")
    assert a == b == "trump wins"


def test_dedup_key_keeps_hashtags():
    assert "#gopdebate" in dedup_key("Watching #GOPDebate")
