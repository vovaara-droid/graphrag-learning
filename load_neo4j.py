"""
Завантажує data/triples.csv у Neo4j.
Для кожної трійки MERGE суб'єкт-вузол, об'єкт-вузол і зв'язок між ними.
"""

import re

import pandas as pd
from neo4j import GraphDatabase

URI = "bolt://localhost:7687"
AUTH = ("neo4j", "graphrag123")
CSV_PATH = "data/triples.csv"


def sanitize_rel_type(predicate: str) -> str:
    """Предикат -> валідний тип зв'язку Neo4j: UPPERCASE, пробіли/спецсимволи -> _"""
    rel = predicate.upper().strip()
    rel = re.sub(r"[^A-ZА-ЯІЇЄҐ0-9]+", "_", rel)
    rel = rel.strip("_")
    return rel or "RELATED_TO"


def load_triples(driver, df: pd.DataFrame) -> tuple[int, int]:
    nodes_created = 0
    rels_created = 0

    with driver.session() as session:
        for _, row in df.iterrows():
            subject = str(row["subject"]).strip()
            predicate = str(row["predicate"]).strip()
            obj = str(row["object"]).strip()
            subject_type = str(row["subject_type"]).strip() or "Entity"
            object_type = str(row["object_type"]).strip() or "Entity"
            source = str(row["source_title"]).strip()

            rel_type = sanitize_rel_type(predicate)

            # MERGE вузлів і зв'язку одним Cypher-запитом
            query = f"""
            MERGE (s:`{subject_type}` {{name: $subject}})
            ON CREATE SET s.created = true
            WITH s, s.created AS s_new
            MERGE (o:`{object_type}` {{name: $obj}})
            ON CREATE SET o.created = true
            WITH s, o, s_new, o.created AS o_new
            MERGE (s)-[r:`{rel_type}` {{source: $source}}]->(o)
            ON CREATE SET r.created = true
            RETURN s_new, o_new, r.created AS r_new
            """

            result = session.run(query, subject=subject, obj=obj, source=source)
            record = result.single()
            if record:
                if record["s_new"]:
                    nodes_created += 1
                if record["o_new"]:
                    nodes_created += 1
                if record["r_new"]:
                    rels_created += 1

    return nodes_created, rels_created


def get_totals(driver) -> tuple[int, int]:
    with driver.session() as session:
        nodes = session.run("MATCH (n) RETURN count(n) AS cnt").single()["cnt"]
        rels = session.run("MATCH ()-[r]->() RETURN count(r) AS cnt").single()["cnt"]
    return nodes, rels


def main() -> None:
    df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")
    print(f"Завантажую {len(df)} трійок з {CSV_PATH}...")

    driver = GraphDatabase.driver(URI, auth=AUTH)
    driver.verify_connectivity()
    print("Підключено до Neo4j.")

    nodes_created, rels_created = load_triples(driver, df)

    total_nodes, total_rels = get_totals(driver)
    driver.close()

    print(f"\nНових вузлів створено:   {nodes_created}")
    print(f"Нових зв'язків створено: {rels_created}")
    print(f"\nУсього у графі:")
    print(f"  Вузлів:  {total_nodes}")
    print(f"  Зв'язків: {total_rels}")


if __name__ == "__main__":
    main()
