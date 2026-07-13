from core.wsi.build_camelyon16_manifest import discover_slides

def test_manifest_function_exists():
    assert callable(discover_slides)
