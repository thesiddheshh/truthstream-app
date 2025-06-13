# tests/test_entity_linking.py

from src.verification.entity_linking import EntityLinker

def test_extract_entities():
    linker = EntityLinker()
    
    text = "Barack Obama visited NASA headquarters in Washington D.C."
    entities = linker.extract_entities(text)
    
    assert isinstance(entities, list)
    assert any(e in entities for e in ["Barack Obama", "Barack"])
    assert "NASA" in entities
    assert any(e in entities for e in ["Washington", "Washington D.C."])

def test_link_known_entities():
    linker = EntityLinker()

    known_names = {
        "Barack Obama": "Q76",
        "NASA": "Q2784",
        "Elon Musk": "Q317521"
    }

    for name, expected_id in known_names.items():
        result = linker.link_entity_to_wikidata(name)
        assert result is not None, f"Failed to link {name}"
        assert result["id"] == expected_id, f"{name} linked to wrong ID: {result['id']}"

def test_link_unknown_entity():
    linker = EntityLinker()
    result = linker.link_entity_to_wikidata("FictionalPersonXYZ")
    assert result is None, "Unknown entity should return None"