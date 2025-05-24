
import pytest
from zeroband.inference.genesys.sanskrit import compute_sanskrit_poetry_reward


class TestSanskritMeters:
    """Tests for identifying various Sanskrit meters."""

    @pytest.mark.parametrize("meter_name,sample_verse", [
        # 8-syllable meters
        ("Anushtubh", "यदा यदा हि धर्मस्य ग्लानिर्भवति भारत।\nअभ्युत्थानमधर्मस्य तदात्मानं सृजाम्यहम्॥"),
        ("Sloka", "धर्मक्षेत्रे कुरुक्षेत्रे समवेता युयुत्सवः।\nमामकाः पाण्डवाश्चैव किमकुर्वत सञ्जय॥"),
        
        # 11-syllable meters
        ("Indravajra", "किं जातु कामस्य कृतेन तेन विद्यासुधास्वादरसं विहाय।\nपुंसां हिताय प्रकटीकृतोऽयं शब्दार्थशास्त्रप्रकटः प्रबन्धः॥"),
        ("Upendravajra", "भजन्ति ये विष्णुपदारविन्दं ते पापिनोऽपि प्रभवन्ति मुक्तैः।\nहिमालयो नाम नगाधिराजो यदाश्रयादुग्रविषोऽपि सर्पः॥"),
        
        # 14-syllable meters
        ("Vasantatilaka", "कश्चित्कान्ताविरहगुरुणा स्वाधिकारात्प्रमत्तः\nशापेनास्तंगमितमहिमा वर्षभोग्येण भर्तुः॥"),
        
        # 17-syllable meters
        ("Mandakranta", "धन्यास्ताः खलु निर्विशङ्कमनसः प्रेमप्रकर्षोत्सुकाः\nकामं स्वैरमुपात्तदानसुलभैरङ्गैर्विलासाकुलाः॥"),
        
        # 19-syllable meters
        ("Sardullavikridita", "यद्गीतं यद्दत्तं यदपि च सुरेभ्यो हुतवहे\nयच्चानिष्ठे दत्तं सुकृतिषु च सर्वं द्विजमुखे॥"),
        
        # 21-syllable meters
        ("Sragdhara", "काव्यालापाँश्च मुञ्च प्रकृतिसुभगान् रोचिषीष्टा यदि त्वं\nशृण्वन्तस्त्वद्गिरं ते मृदुपदरचनामेव साधु प्रशंसुः॥")
    ])
    def test_meter_identification(self, meter_name, sample_verse):
        """Test identification of various Sanskrit meters."""
        completion = f"<think>Composing in {meter_name}</think>\n{sample_verse}"
        verification_info = {"meter_type": meter_name, "topic": "test"}
        reward = compute_sanskrit_poetry_reward(completion, verification_info)
        assert reward > 0.35, f"Failed to identify {meter_name} meter"


class TestErrorHandling:
    """Tests for error handling and edge cases."""
    
    def test_empty_text(self):
        """Test with empty text."""
        completion = "<think>Empty text</think>\n"
        verification_info = {"meter_type": "Sloka", "topic": "empty"}
        reward = compute_sanskrit_poetry_reward(completion, verification_info)
        assert reward == 0.0
    
    def test_non_sanskrit_text(self):
        """Test with non-Sanskrit text."""
        completion = "<think>Non-Sanskrit</think>\nThis is English text only. No Sanskrit here."
        verification_info = {"meter_type": "Sloka", "topic": "invalid"}
        reward = compute_sanskrit_poetry_reward(completion, verification_info)
        assert reward == 0.0
    
    def test_too_short_text(self):
        """Test with text that's too short."""
        completion = "<think>Short</think>\nयदा यदा"
        verification_info = {"meter_type": "Sloka", "topic": "short"}
        reward = compute_sanskrit_poetry_reward(completion, verification_info)
        assert reward == 0.0
    
    def test_mixed_script_text(self):
        """Test with mixed script text."""
        completion = "<think>Mixed script</think>\nयदा यदा हि धर्मस्य This is mixed with English."
        verification_info = {"meter_type": "Sloka", "topic": "mixed"}
        reward = compute_sanskrit_poetry_reward(completion, verification_info)
        # Should still attempt to analyze the Sanskrit portions
        # But likely with low reward due to pattern mismatch
        assert reward <= 0.4
    
    def test_anti_gaming(self):
        """Test that repetitive patterns are rejected."""
        completion = """<think>Gaming the system</think>
ल ल ल ल ल ल ल ल
ल ल ल ल ल ल ल ल"""
        
        verification_info = {"meter_type": "Sloka", "topic": "test"}
        reward = compute_sanskrit_poetry_reward(completion, verification_info)
        assert reward == 0.0  # Should reject repetitive pattern


class TestMeterVariations:
    """Tests for meter name variations and spelling."""
    
    @pytest.mark.parametrize("meter_input,expected_meter", [
        ("sloka", "Sloka"),
        ("Sloka", "Sloka"),
        ("anushtubh", "Anushtubh"),
        ("Anushtubh", "Anushtubh"),
        ("sardullavikridita", "Sardullavikridita"),
        ("Mandakranta", "Mandakranta"),
        ("vasantatilaka", "Vasantatilaka")
    ])
    def test_meter_name_variations(self, meter_input, expected_meter):
        """Test that various spellings of meter names are recognized."""
        # Using a standard Sloka verse for simplicity
        sample_verse = "धर्मक्षेत्रे कुरुक्षेत्रे समवेता युयुत्सवः।\nमामकाः पाण्डवाश्चैव किमकुर्वत सञ्जय॥"
        completion = f"<think>Composing in {meter_input}</think>\n{sample_verse}"
        
        # Use the input meter name in verification
        verification_info = {"meter_type": meter_input, "topic": "test"}
        reward = compute_sanskrit_poetry_reward(completion, verification_info)
        
        # First, check if the meter is in the database using the normalized version
        from zeroband.inference.genesys.sanskrit import SanskritMeterDatabase
        meter_info = SanskritMeterDatabase.get_meter_info(meter_input)
        assert meter_info is not None, f"Meter name '{meter_input}' not recognized by the database"
        
        # Now check the reward
        assert reward > 0, f"Failed to recognize meter name variation: {meter_input}"
        
    def test_unicode_normalization(self):
        """Test separate test case for Unicode normalization with diacritical marks."""
        # This test checks if the SanskritMeterDatabase.get_meter_info method works
        # It doesn't test the full reward computation which relies on chandas
        from zeroband.inference.genesys.sanskrit import SanskritMeterDatabase
        
        # Test with diacritical marks
        unicode_variations = [
            ("Śloka", "sloka"),
            ("śloka", "sloka"),
            ("Anuṣṭubh", "anushtubh"),
            ("Śārdūlavikrīḍita", "sardullavikridita"),
            ("Mandākrāntā", "mandakranta"),
            ("Vasanta-tilakā", "vasantatilaka")
        ]
        
        for unicode_name, normalized_name in unicode_variations:
            meter_info = SanskritMeterDatabase.get_meter_info(unicode_name)
            assert meter_info is not None, f"Failed to normalize Unicode meter name: {unicode_name}"
            
            # Verify we get the same meter info with both forms
            standard_info = SanskritMeterDatabase.get_meter_info(normalized_name)
            assert meter_info == standard_info, f"Normalized {unicode_name} doesn't match {normalized_name}"


class TestEdgeCases:
    """Tests for edge cases and special scenarios."""
    
    def test_unknown_meter(self):
        """Test with a meter not in the database."""
        completion = "<think>Unknown meter</think>\nयदा यदा हि धर्मस्य ग्लानिर्भवति भारत।"
        verification_info = {"meter_type": "NonExistentMeter", "topic": "test"}
        reward = compute_sanskrit_poetry_reward(completion, verification_info)
        assert reward == 0.0
    
    def test_with_punctuation(self):
        """Test with text containing punctuation."""
        completion = "<think>With punctuation</think>\nयदा यदा हि धर्मस्य, ग्लानिर्भवति भारत!\nअभ्युत्थानमधर्मस्य? तदात्मानं सृजाम्यहम्॥"
        verification_info = {"meter_type": "Sloka", "topic": "test"}
        reward = compute_sanskrit_poetry_reward(completion, verification_info)
        # Should be able to handle punctuation
        assert reward > 0.3
    
    def test_with_extra_whitespace(self):
        """Test with text containing extra whitespace."""
        completion = "<think>Extra whitespace</think>\n  यदा यदा हि धर्मस्य  \n\n  ग्लानिर्भवति भारत  "
        verification_info = {"meter_type": "Sloka", "topic": "test"}
        reward = compute_sanskrit_poetry_reward(completion, verification_info)
        # Should normalize whitespace
        assert reward > 0.3


class TestPerformance:
    """Tests for system performance with longer texts."""
    
    def test_longer_poem(self):
        """Test with a longer multi-verse poem."""
        # Bhagavad Gita excerpt (4 verses)
        completion = """<think>Longer poem in Sloka meter</think>
धर्मक्षेत्रे कुरुक्षेत्रे समवेता युयुत्सवः।
मामकाः पाण्डवाश्चैव किमकुर्वत सञ्जय॥

सञ्जय उवाच।
दृष्ट्वा तु पाण्डवानीकं व्यूढं दुर्योधनस्तदा।
आचार्यमुपसङ्गम्य राजा वचनमब्रवीत्॥

पश्यैतां पाण्डुपुत्राणामाचार्य महतीं चमूम्।
व्यूढां द्रुपदपुत्रेण तव शिष्येण धीमता॥

अत्र शूरा महेष्वासा भीमार्जुनसमा युधि।
युयुधानो विराटश्च द्रुपदश्च महारथः॥"""
        
        verification_info = {"meter_type": "Sloka", "topic": "Mahabharata"}
        reward = compute_sanskrit_poetry_reward(completion, verification_info)
        assert reward > 0.3, "Failed with longer multi-verse poem"


class TestChandasDependency:
    """Tests for system behavior when chandas library is unavailable."""
    
    def test_chandas_unavailable(self, monkeypatch):
        """Test behavior when chandas library is unavailable."""
        # Mock an import error for chandas modules
        def mock_import_error(*args, **kwargs):
            raise ImportError("Mocked chandas import error")
        
        # Apply the mock to the identify.to_pattern_lines function
        from zeroband.inference.genesys.sanskrit import SanskritProsodyAnalyzer
        monkeypatch.setattr(SanskritProsodyAnalyzer, "extract_prosodic_patterns", mock_import_error)
        
        # Try to compute reward
        completion = "<think>Test</think>\nयदा यदा हि धर्मस्य ग्लानिर्भवति भारत।"
        verification_info = {"meter_type": "Sloka", "topic": "test"}
        reward = compute_sanskrit_poetry_reward(completion, verification_info)
        
        # Should return 0.0 when chandas is unavailable
        assert reward == 0.0


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])