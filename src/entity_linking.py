# src/verification/entity_linking.py

import spacy
from SPARQLWrapper import SPARQLWrapper, JSON
import logging
import re

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

class EntityLinker:
    def __init__(self):
        """Initializes linker with Wikidata SPARQL client"""
        self.sparql = SPARQLWrapper("https://query.wikidata.org/sparql") 
        self.known_entities = {
            "Barack Obama": "Q76",
            "NASA": "Q2784",
            "Elon Musk": "Q317521",
            "Bill Gates": "Q3384",
            "World Health Organization": "Q1103"
        }

    def extract_entities(self, text):
        """
        Extracts named entities using spaCy.
        
        Args:
            text (str): Input article/post text
            
        Returns:
            list: List of extracted entity names (deduped and normalized)
        """
        doc = nlp(text)
        raw_entities = [ent.text for ent in doc.ents if ent.label_ in ["PERSON", "ORG", "GPE"]]

        # Normalize full phrases like "Washington D.C." → "Washington"
        normalized = []
        for name in raw_entities:
            base_name = re.sub(r'\s+D\.C\.$', '', name)  # Remove "D.C." suffix
            base_name = re.sub(r'\s*\([^)]*\)', '', base_name)  # Remove parentheticals
            normalized.append(base_name)

        return list(set(normalized))  # Deduplicate

    def link_entity_to_wikidata(self, name):
        """
        Tries to find Wikidata ID for a given name.
        
        Args:
            name (str): Name of the entity
            
        Returns:
            dict: {"name": str, "id": str} if found
        """
        # Try known entities first
        if name in self.known_entities:
            return {"name": name, "id": self.known_entities[name]}

        # Then try Wikidata lookup
        try:
            query = f"""
            SELECT ?item ?itemLabel WHERE {{
              ?item rdfs:label "{name}"@en.
              SERVICE wikibase:label {{ bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }}
            }} LIMIT 1
            """
            self.sparql.setQuery(query)
            self.sparql.setReturnFormat(JSON)
            results = self.sparql.query().convert()

            if results["results"]["bindings"]:
                item_url = results["results"]["bindings"][0]["item"]["value"]
                entity_id = item_url.split("/")[-1]
                label = results["results"]["bindings"][0]["itemLabel"]["value"]
                logger.info(f"Found Wikidata match for '{name}' → {entity_id}")
                return {"name": label, "id": entity_id}
            else:
                logger.warning(f"No Wikidata match found for '{name}'")
                return None
        except Exception as e:
            logger.error(f"Wikidata lookup failed for '{name}': {e}")
            return None

if __name__ == "__main__":
    linker = EntityLinker()
    sample_text = "Barack Obama visited NASA headquarters in Washington D.C."
    entities = linker.extract_entities(sample_text)
    print("Extracted Entities:", entities)

    for entity in entities:
        result = linker.link_entity_to_wikidata(entity)
        if result:
            print("Linked Entity:", result)