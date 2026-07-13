import openslide

def test_openslide_import():
    assert openslide is not None
    assert hasattr(openslide, "OpenSlide")
