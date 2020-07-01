#!/usr/bin/env python
from neo4j import GraphDatabase
from sklearn.cluster import KMeans
import numpy as np

NUM_CLUSTERS=6
HOST = "192.168.1.167"
driver = GraphDatabase.driver("bolt://" + HOST + ":7687", auth=("neo4j", "password"))

NODE2VEC = """
CALL gds.alpha.node2vec.stream({
  nodeProjection: 'Character',
  relationshipProjection: 'APPEARED_WITH',
  dimensions: 16,
  returnFactor: 1.0,
  inOutFactor: 2.0
}) YIELD nodeId, embedding
"""

print("Generating graph embeddings...")
embeddings = []
with driver.session() as session:
    results = session.run(NODE2VEC)
    for result in results:
        embeddings.append(result)
print("...generated {} embeddings".format(len(embeddings)))

print("Performing K-Means clustering (n_clusters={})...".format(NUM_CLUSTERS))
X = np.array([e["embedding"] for e in embeddings])
kmeans = KMeans(n_clusters=NUM_CLUSTERS).fit(X)
clustering = []
for idx, cluster in enumerate(kmeans.predict(X)):
    clustering.append({"nodeId": embeddings[idx]["nodeId"], "clusterId": int(cluster)})
print("...clustering completed.")

UPDATE = """
UNWIND $updates AS update
    MATCH (n) WHERE id(n) = update.nodeId
    SET n.clusterId = update.clusterId
"""

def update_tx(tx, cypher, **kwargs):
    result = tx.run(cypher, kwargs)
    return result.consume()

print("Updating graph...")
with driver.session() as session:
    result = session.write_transaction(update_tx, UPDATE, updates=clustering)
    print("...update complete: {}".format(result.counters))
