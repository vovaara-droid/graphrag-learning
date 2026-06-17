import json
from neo4j import GraphDatabase

driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "graphrag123"))
with driver.session() as s:
    nodes = s.run("MATCH (n) RETURN count(n) AS c").single()["c"]
    rels = s.run("MATCH ()-[r]->() RETURN count(r) AS c").single()["c"]
    labels = [r["label"] for r in s.run("CALL db.labels() YIELD label RETURN label ORDER BY label").data()]
    rel_types = [r["relationshipType"] for r in s.run("CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType ORDER BY relationshipType").data()]
    samples = {}
    for lbl in labels:
        rows = s.run(f"MATCH (n:`{lbl}`) RETURN n.name AS name LIMIT 5").data()
        samples[lbl] = [r["name"] for r in rows]
driver.close()

result = {
    "nodes_total": nodes,
    "rels_total": rels,
    "labels": labels,
    "rel_types_count": len(rel_types),
    "rel_types_sample": rel_types[:15],
    "node_samples": samples,
}

with open("neo4j_stats.json", "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)

print("done")
