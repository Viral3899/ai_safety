from src.models.abuse_detector import AbuseDetector


def test_abuse_detector_basic():
    d = AbuseDetector(threshold=0.5)
    res = d.predict('I hate you')
    assert res.label == 'abuse'
