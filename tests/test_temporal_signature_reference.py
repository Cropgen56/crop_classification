from src.labeling.temporal_reference import canonicalize_crop_name, load_crop_signature_reference


def test_canonicalize_crop_name_normalizes_aliases():
    assert canonicalize_crop_name("Paddy / rice") == "paddy"
    assert canonicalize_crop_name("Gram / chickpea") == "gram"
    assert canonicalize_crop_name("Tur / arhar / pigeon pea") == "tur"
    assert canonicalize_crop_name("Jowar / sorghum") == "jowar"


def test_load_crop_signature_reference_includes_expected_crop_templates():
    refs = load_crop_signature_reference()
    crops = {row["crop"] for row in refs}
    assert "paddy" in crops
    assert "soybean" in crops
    assert "cotton" in crops
    assert "sugarcane" in crops
    assert "wheat" in crops
