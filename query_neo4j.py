import json
from neo4j import GraphDatabase

driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "graphrag123"))

queries = {
    "top_subjects": """
        MATCH (n)-[r]->()
        RETURN n.name AS name, labels(n)[0] AS type, count(r) AS actions
        ORDER BY actions DESC LIMIT 10
    """,
    "russian_troops": """
        MATCH (n {name: "російські війська"})-[r]->(m)
        RETURN n.name AS subject, type(r) AS predicate, m.name AS object
    """,
    "attacks": """
        MATCH (n)-[r:АТАКУВАЛИ]->(m)
        RETURN n.name AS who, m.name AS what
    """,
    "top_locations": """
        MATCH ()-[r]->(m:LOC)
        RETURN m.name AS location, count(r) AS mentions
        ORDER BY mentions DESC LIMIT 10
    """,
}

results = {}
with driver.session() as s:
    for key, q in queries.items():
        results[key] = s.run(q).data()

driver.close()

with open("neo4j_queries.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print("done")
