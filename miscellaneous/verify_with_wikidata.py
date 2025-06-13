# src/verification/verify_with_wikidata.py

from SPARQLWrapper import SPARQLWrapper, JSON
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WikidataFactChecker:
    def __init__(self):
        """Initializes connection to Wikidata SPARQL endpoint"""
        self.endpoint_url = "https://query.wikidata.org/sparql" 
        self.sparql = SPARQLWrapper(self.endpoint_url)

    def _run_query(self, query):
        """
        Executes a SPARQL query and returns results.
        
        Args:
            query (str): Valid SPARQL query string
            
        Returns:
            dict: JSON response from Wikidata
        """
        try:
            self.sparql.setQuery(query)
            self.sparql.setReturnFormat(JSON)
            results = self.sparql.query().convert()
            return results["results"]["bindings"]
        except Exception as e:
            logger.error(f"SPARQL query failed: {e}")
            return []

    def verify_claim(self, subject_id, property_id, object_id):
        """
        Verifies if a triple (subject, property, object) exists in Wikidata.
        
        Args:
            subject_id (str): Wikidata ID of subject (e.g., Q76)
            property_id (str): Wikidata property ID (e.g., P106)
            object_id (str): Wikidata ID of object (e.g., Q11696)
            
        Returns:
            bool: True if claim exists
        """
        query = f"""
        ASK {{
          wd:{subject_id} wdt:{property_id} wd:{object_id}.
        }}
        """
        try:
            self.sparql.setQuery(query)
            self.sparql.setReturnFormat(JSON)
            result = self.sparql.query().convert()
            return result['boolean']
        except Exception as e:
            logger.error(f"Verification error: {e}")
            return False

    def verify_article_entities(self, entities, claims=None):
        """
        Verify article content using linked entities and known claims.
        
        Args:
            entities (dict): Mapped entities like {"Bill Gates": "Q3384", ...}
            claims (list): List of (subject, predicate, object) triples to verify
            
        Returns:
            list: Verified claims or None if no match
        """
        if not claims:
            # Default test claim: check if Bill Gates is an instance of human
            claims = [("Q3384", "P31", "Q5")]  # Bill Gates → instance of → Human

        verified_claims = []
        for claim in claims:
            subject, predicate, obj = claim
            result = self.verify_claim(subject, predicate, obj)
            verified_claims.append({
                "claim": claim,
                "verified": result
            })
            logger.info(f"Claim {claim} → {'Verified' if result else 'Unverified'}")
        return verified_claims