#!/usr/bin/env python
import getopt, sys
from neo4j import GraphDatabase
from sklearn.cluster import KMeans
import numpy as np

DEFAULT_URI = "bolt://localhost:7687"
DEFAULT_USER = "neo4j"
DEFAULT_PASS = "password"
NUM_CLUSTERS=6

NODE2VEC_CYPHER = """
CALL gds.alpha.node2vec.stream({
  nodeProjection: 'Character',
  relationshipProjection: 'APPEARED_WITH',
  dimensions: $d,
  returnFactor: $p,
  inOutFactor: $q
}) YIELD nodeId, embedding
"""

UPDATE_CYPHER = """
UNWIND $updates AS update
    MATCH (n) WHERE id(n) = update.nodeId
    SET n.clusterId = update.clusterId
"""

def usage():
    print("usage:\t kmeans.py [-A BOLT URI] [-U USERNAME (default: neo4j)] [-P PASSWORD (default: password)]")
    print("supported parameters:")
    print("\t-d DIMENSIONS (default: 16)")
    print("\t-p RETURN PARAMETER (default: 1.0)")
    print("\t-q IN-OUT PARAMETER (default: 1.0)")
    print("\t-k K-MEANS NUM_CLUSTERS (default: 6)")
    sys.exit(1)

def extract_embeddings(driver, p=1.0, q=1.0, d=16):
    """
    Call the GDS neo2vec routine using the given driver and provided params.
    """
    print("Generating graph embeddings (p={}, q={}, d={})...".format(p, q, d))
    embeddings = []
    with driver.session() as session:
        results = session.run(NODE2VEC_CYPHER, p=float(p), q=float(q), d=int(d))
        for result in results:
            embeddings.append(result)
    print("...generated {} embeddings".format(len(embeddings)))
    return embeddings


def kmeans(embeddings, k=NUM_CLUSTERS):
    """
    Given a list of dicts like {"nodeId" 1, "embedding": [1.0, 0.1, ...]},
    generate a list of dicts like {"nodeId": 1, "clusterId": 2}
    """
    print("Performing K-Means clustering (n_clusters={})...".format(NUM_CLUSTERS))
    X = np.array([e["embedding"] for e in embeddings])
    kmeans = KMeans(n_clusters=int(k)).fit(X)
    clustering = []
    for idx, cluster in enumerate(kmeans.predict(X)):
        clustering.append({"nodeId": embeddings[idx]["nodeId"], "clusterId": int(cluster)})
    print("...clustering completed.")
    return clustering


def _update_tx(tx, cypher, **kwargs):
    result = tx.run(cypher, kwargs)
    return result.consume()

def update_clusters(driver, clusters):
    """
    Given a list of dicts with "nodeId" and "clusterId", update the graph by
    setting the "clusterId" property on each node.
    """
    print("Updating graph...")
    with driver.session() as session:
        result = session.write_transaction(_update_tx, UPDATE_CYPHER, updates=clusters)
        print("...update complete: {}".format(result.counters))


if __name__ == '__main__':
    # getopt, because: "POSIX getopt(1) is The Correct Way" ~sircmpwn
    try:
        opts, args = getopt.getopt(sys.argv[1:], "hA:U:P:p:q:d:k:")
    except getopt.GetoptError as err:
        print(err)
        usage()

    uri = DEFAULT_URI
    user = DEFAULT_USER
    password = DEFAULT_PASS
    p = 1.0
    q = 1.0
    d = 16
    k = 6

    for o, a in opts:
        if o == "-h":
            usage()
        elif o == "-A":
            uri = a
        elif o == "-U":
            user = a
        elif o == "-P":
            password = a
        elif o == "-p":
            p = a
        elif o == "-q":
            q = a
        elif o == "-d":
            d = a
        elif o == "-k":
            k = a
        else:
            usage()

    print("Connecting to uri: {}".format(uri))
    driver = GraphDatabase.driver(uri, auth=(user, password))
    embeddings = extract_embeddings(driver, p=p, q=q, d=d)
    clusters = kmeans(embeddings, k=k)
    update_clusters(driver, clusters)
    driver.close()
