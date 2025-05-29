import pytest

from zeroband.inference.genesys.sanskrit_morphology import compute_sanskrit_morphology_reward

# --- Test Data ---
@pytest.fixture
def sample_verification_info():
    # Bhvādi dhātu -> present (Laṭ) active 1sg"
    return {
        "dhatu": "BU",
        "gana": "BvAdi", 
        "lakara": "la~w",
        "prayoga": "kartari",
        "purusha": "uttama",
        "vacana": "eka",
    }

# --- Test Reward Correctness ---
class TestRewardCorrectness:
    """Test the core reward computation logic for correct, incorrect, and partially correct answers."""
    
    # Correct answer cases
    FULL_REWARD_CASES = [
        (
            "Bhvādi dhātu -> present (Laṭ) active 3sg",
            {
                "dhatu": "BU",  # √भू 'to be, become'
                "gana": "BvAdi",
                "lakara": "la~w",
                "prayoga": "kartari",
                "purusha": "praTama",
                "vacana": "eka",
            },
            "[[Bavati]]", # correct form (भवति)
        ),
        (
            "Adādi dhātu -> perfect past (Liṭ) passive 1du",
            {
                "dhatu": "dA",  # √दा 'to give'
                "gana": "adAdi",
                "lakara": "li~w",
                "prayoga": "karmaRi",
                "purusha": "uttama",
                "vacana": "dvi",
            },
            "[[dadivahe]]", # correct form (ददिवहे)
        ),
        (
            "Juhotyādi ubhayapadī dhātu -> periphrastic future (Luṭ) active 3pl",
            {
                "dhatu": "pF",  # √पॄ 'to fill'
                "gana": "juhotyAdi",
                "lakara": "lu~w",
                "prayoga": "kartari",
                "purusha": "praTama",
                "vacana": "bahu",
            },
            "[[parItAraH]]", # correct parasmaipada form (परीतारः)
        )
    ]
    
    @pytest.mark.parametrize("case_description,verification_info,completion", FULL_REWARD_CASES)
    def test_full_reward_for_correct_answers(self, case_description, verification_info, completion):
        """Test that correct Sanskrit forms receive full reward (1.0)."""
        result = compute_sanskrit_morphology_reward(completion, verification_info)
        assert result == 1.0, f"Failed for case: {case_description}"
    
    # Incorrect answer cases
    NO_REWARD_CASES = [
        (
            "Divādi ubhayapadī dhātu -> simple future (Lṛṭ) active 3pl",
            {
                "dhatu": "jFz",  # √जॄष् 'to age'
                "gana": "divAdi",
                "lakara": "lf~w",
                "prayoga": "kartari",
                "purusha": "praTama",
                "vacana": "bahu",
            },
            "[[jFzyanti]]", # incorrect form (जॄष्यन्ति); correct forms: parasmaipada jarIzyanti (जरीष्यन्ति) or ātmanepada jarizyanti (जरिष्यन्ति)
        ),
        (
            "Svādi ubhayapadī dhātu -> Vedic subjunctive (Leṭ) active 3sg",
            {
                "dhatu": "vfY",  # √वृञ् 'to choose'
                "gana": "svAdi",
                "lakara": "le~w",
                "prayoga": "kartari",
                "purusha": "praTama",
                "vacana": "eka",
            },
            "[[varaYAte]]", # incorrect form (वरञाते); correct forms: parasmaipada vfRuvAte (वृणुवाते) or ātmanepada vfRuvAti (वृणुवाति)
        ),
        (
            "Tudādi dhātu -> imperative (Loṭ) active 1sg",
            {
                "dhatu": "RU",  # √णू 'to praise'
                "gana": "tudAdi",
                "lakara": "lo~w",
                "prayoga": "kartari",
                "purusha": "uttama",
                "vacana": "eka",
            },
            "[[nUvani]]", # incorrect form (नूवनि); correct form: nuvAni (नुवानि) 
        ),
    ]
    
    @pytest.mark.parametrize("case_description,verification_info,completion", NO_REWARD_CASES)
    def test_no_reward_for_incorrect_answers(self, case_description, verification_info, completion):
        """Test that incorrect answers receive no reward (0.0)."""
        result = compute_sanskrit_morphology_reward(completion, verification_info)
        assert result == 0.0, f"Failed for case: {case_description}"

    # Partially correct answer cases
    """
    Correct derivation of apfcyAvahi (अपृच्यावहि):
    ===================
    1.3.1     : pfcI~
    1.3.2     : pfcI~
    1.3.9     : pfc
    3.2.111   : pfc + laN
    1.3.3     : pfc + laN
    1.3.9     : pfc + la
    1.3.13    : pfc + la
    3.4.78    : pfc + vahi
    3.4.113   : pfc + vahi
    3.1.67    : pfc + yak + vahi
    1.3.3     : pfc + yak + vahi
    1.3.9     : pfc + ya + vahi
    3.4.114   : pfc + ya + vahi
    1.2.4     : pfc + ya + vahi
    1.4.13    : pfc + ya + vahi <---------- partial credit (15/22)
    6.4.71    : aw + pfc + ya + vahi
    1.3.3     : aw + pfc + ya + vahi
    1.3.9     : a + pfc + ya + vahi
    1.1.5     : a + pfc + ya + vahi
    1.4.14    : a + pfc + ya + vahi
    7.3.101   : a + pfc + yA + vahi
    8.4.68    : a + pfc + yA + vahi
    """
    apṛcyāvahi_case = (
        "Rudhādi dhātu -> imperfect (Laṅ) passive 1du",
        {
            "dhatu": "pfcI~",  # √पृच् 'to mix'
            "gana": "ruDAdi",
            "lakara": "la~N",
            "prayoga": "karmaRi",
            "purusha": "uttama",
            "vacana": "dvi",
        },
        "[[pfcyavahi]]", # partially correct form (पृच्यावहि); correct form: apfcyAvahi (अपृच्यावहि)
        0.68, # rounded to 2 decimal places
    )

    """
    Correct derivation of kriyAstam (क्रियास्तम्):
    ===================
    1.3.1     : kf
    3.3.173   : kf + li~N
    1.3.2     : kf + li~N
    1.3.3     : kf + li~N
    1.3.9     : kf + l
    1.3.78    : kf + l
    3.4.78    : kf + Tas
    1.3.4     : kf + Tas
    3.4.116   : kf + Tas
    3.4.101   : kf + tam
    1.3.4     : kf + tam <---------- partial credit (11/23)
    3.4.104   : kf + yAsu~w + tam
    1.3.2     : kf + yAsu~w + tam
    1.3.3     : kf + yAsu~w + tam
    1.3.9     : kf + yAs + tam
    3.4.107   : kf + yAs + stam
    3.4.116   : kf + yAs + stam
    1.4.13    : kf + yAs + stam
    1.1.5     : kf + yAs + stam
    7.4.28    : kri + yAs + stam
    1.4.14    : kri + yAs + stam
    8.2.29    : kri + yA + stam
    8.4.68    : kri + yA + stam
    """
    kriyāstam_case = (
        "Tanādi dhātu -> benedictive (Aśir-Liṅ) active 2du",
        {
            "dhatu": "kf",  # √कृ 'to do'
            "gana": "tanAdi",
            "lakara": "ASIrli~N",
            "prayoga": "kartari",
            "purusha": "maDyama",
            "vacana": "dvi",
        },
        "[[kftam]]", # partially correct form (कृतम्); correct form: kriyAstam (क्रियास्तम्)
        0.48, # rounded to 2 decimal places
    )

    """
    Correct derivation of Baveta (भवेत):
    ===================
    1.3.1     : BU
    3.3.161   : BU + li~N
    1.3.2     : BU + li~N
    1.3.3     : BU + li~N
    1.3.9     : BU + l
    1.3.78    : BU + l
    3.4.78    : BU + Ta
    3.4.113   : BU + Ta <---------- partial credit (8/29)
    3.1.68    : BU + Sap + Ta
    1.3.3     : BU + Sap + Ta
    1.3.8     : BU + Sap + Ta
    1.3.9     : BU + a + Ta
    3.4.113   : BU + a + Ta
    3.4.101   : BU + a + ta
    3.4.103   : BU + a + yAsu~w + ta
    1.3.2     : BU + a + yAsu~w + ta
    1.3.3     : BU + a + yAsu~w + ta
    1.3.9     : BU + a + yAs + ta
    3.4.107   : BU + a + yAs + sta
    1.2.4     : BU + a + yAs + sta
    1.4.13    : BU + a + yAs + sta
    7.2.79    : BU + a + yA + ta
    7.2.80    : BU + a + iy + ta
    6.1.66    : BU + a + i + ta
    7.3.84    : Bo + a + i + ta
    1.4.14    : Bo + a + i + ta
    6.1.78    : Bav + a + i + ta
    6.1.87    : Bav +  + e + ta
    8.4.68    : Bav +  + e + ta
    """
    bhaveta_case = (
        "Bhvādi dhātu -> optative (Vidhi-Liṅ) active 2pl",
        {
            "dhatu": "BU",  # √भू 'to be, become'
            "gana": "BvAdi",
            "lakara": "viDili~N",
            "prayoga": "kartari",
            "purusha": "maDyama",
            "vacana": "bahu",
        },
        "[[BUTa]]", # partially correct form (भूथ); correct form: Baveta (भवेत)
        0.28, # rounded to 2 decimal places
    )

    PARTIAL_REWARD_CASES = [apṛcyāvahi_case, kriyāstam_case, bhaveta_case]
    
    @pytest.mark.parametrize("case_description,verification_info,completion,expected_reward", PARTIAL_REWARD_CASES)
    def test_partial_reward_for_partially_correct_answer(self, case_description, verification_info, completion, expected_reward):
        """Test cases with specific expected partial scores."""
        result = compute_sanskrit_morphology_reward(completion, verification_info)
        rounded_result = round(result, 2)
        assert rounded_result == expected_reward, f"Failed for case: {case_description}, expected {expected_reward}, got {result}"


# --- Test Input Handling ---
class TestInputHandling:
    """Test edge cases around input formats, positions, and transliteration schemes."""
    
    # Answer position cases
    ANSWER_POSITION_CASES = [
        ("[[BavAmi]] is the correct form according to Panini.", "Answer at beginning"),
        ("The correct form [[BavAmi]] follows these rules.", "Answer in middle"),
        ("According to Panini, the correct form is [[BavAmi]]", "Answer at end"),
        ("First [[wrong]] then [[BavAmi]] is correct.", "Multiple brackets - takes last"),
    ]
    
    @pytest.mark.parametrize("completion,case_description", ANSWER_POSITION_CASES)
    def test_answer_positions_within_completions(self, completion, case_description, sample_verification_info):
        """Test that answer position within completion doesn't affect correctness."""
        result = compute_sanskrit_morphology_reward(completion, sample_verification_info)
        assert result == 1.0, f"Failed for case: {case_description}"
    
    # Answer whitespace cases
    WHITESPACE_CASES = [
        ("[[ BavAmi ]]", "Whitespace before and after answer", 1.0),
        ("[[Bav Ami]]", "Whitespace within answer", 0.0),
    ]
    
    @pytest.mark.parametrize("completion,case_description,expected_reward", WHITESPACE_CASES)
    def test_answers_with_whitespaces(self, completion, case_description, expected_reward, sample_verification_info):
        """Test that extra whitespace around answers is handled correctly."""
        result = compute_sanskrit_morphology_reward(completion, sample_verification_info)
        assert result == expected_reward, f"Failed for case: {case_description}"
    
    # Transliteration scheme test cases
    TRANSLITERATION_CASES = [
        ("[[BavAmi]]", "SLP1", 1.0),
        ("[[bhavāmi]]", "IAST", 1.0),
        ("[[bhavAmi]]", "Harvard-Kyoto", 1.0),
        ("[[bhavaami]]", "Velthuis", 1.0),
        ("[[भवामि]]", "Devanagari", 1.0),
        ("[[భవామి]]", "Telugu", 1.0),
        ("[[ଭଵାମି]]", "Odia", 1.0),
        ("[[ભવામિ]]", "Gujarati", 1.0),
        ("[[ภวามิ]]", "Thai", 1.0),
        ("[[ಭವಾಮಿ]]", "Kannada", 1.0),
        ("[[ഭവാമി]]", "Malayalam", 1.0),
        ("[[𑖥𑖪𑖯𑖦𑖰]]", "Siddham", 1.0),
        ("[[𑀪𑀯𑀸𑀫𑀺]]", "Brahmi", 1.0),
        ("[[ភវាមិ]]", "Khmer", 1.0),
        ("[[ᏥᎨᏎᏍᏗ]]", "Unsupported scheme (Cherokee)", 0.0),
    ]
    
    @pytest.mark.parametrize("completion,case_description,expected_reward", TRANSLITERATION_CASES)
    def test_answers_with_multiple_transliteration_schemes(self, completion, case_description, expected_reward, sample_verification_info):
        """Test that different transliteration schemes are handled correctly."""
        result = compute_sanskrit_morphology_reward(completion, sample_verification_info)
        assert result == expected_reward, f"Failed for case: {case_description}"
    
    def test_very_long_completion(self, sample_verification_info):
        """Test handling of very long completion text."""
        long_text = "This is a very long completion. " * 100
        completion = f"{long_text} The answer is [[BavAmi]]"
        
        result = compute_sanskrit_morphology_reward(completion, sample_verification_info)
        assert result == 1.0