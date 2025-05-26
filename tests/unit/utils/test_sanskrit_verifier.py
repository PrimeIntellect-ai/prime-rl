import pytest, textwrap
from zeroband.training.sanskrit.verifier import verify_meter

GOOD_ANUSHTUPH = textwrap.dedent("""
    धर्मो रक्षति रक्षितः
    धर्मो हन्ति निहन्यते
    """)  # quick fake example; should scan as Anuṣṭubh

BAD_ANUSHTUPH = "गा गा गा\nगा"

def test_passes_correct_meter():
    result = verify_meter(GOOD_ANUSHTUPH, "Anuṣṭup (Śloka)", strict_match=False)
    print("\nDetected meters:", result)
    assert result["matches_expected"] == True

def test_fails_wrong_meter():
    result = verify_meter(BAD_ANUSHTUPH, "Anuṣṭup (Śloka)")
    assert result["matches_expected"] == False
