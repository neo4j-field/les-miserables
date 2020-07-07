#!/usr/bin/env python
import getopt, sys
from neo4j import GraphDatabase
from sklearn.cluster import KMeans
import numpy as np

DEFAULT_URI = "bolt://localhost:7687"
DEFAULT_USER = "neo4j"
DEFAULT_PASS = "password"
DEFAULT_REL = "UNWEIGHTED_APPEARED_WITH"
DEFAULT_LABEL = "Character"
NUM_CLUSTERS=6

NODE2VEC_CYPHER = """
CALL gds.alpha.node2vec.stream({
  nodeProjection: $L,
  relationshipProjection: {
    EDGE: {
      type: $R,
      orientation: 'UNDIRECTED'
    }
  },
  embeddingSize: $d,
  returnFactor: $p,
  inOutFactor: $q
}) YIELD nodeId, embedding
"""

UPDATE_CYPHER = """
UNWIND $updates AS updateMap
    MATCH (n) WHERE id(n) = updateMap.nodeId
    SET n += updateMap.valueMap
"""

def usage():
    print("usage:\t kmeans.py [-A BOLT URI] [-U USERNAME (default: neo4j)] [-P PASSWORD (default: password)]")
    print("supported parameters:")
    print("\t-R RELATIONSHIP_TYPE (default: 'UNWEIGHTED_APPEARED_WITH'")
    print("\t-L NODE_LABEL (default: 'Character'")
    print("\t-C CLUSTER_PROPERTY (default: 'clusterId'")
    print("\t-d DIMENSIONS (default: 16)")
    print("\t-p RETURN PARAMETER (default: 1.0)")
    print("\t-q IN-OUT PARAMETER (default: 1.0)")
    print("\t-k K-MEANS NUM_CLUSTERS (default: 6)")
    sys.exit(1)

def extract_embeddings(driver, label=DEFAULT_LABEL, relType=DEFAULT_REL,
                       p=1.0, q=1.0, d=16):
    """
    Call the GDS neo2vec routine using the given driver and provided params.
    """
    print("Generating graph embeddings (p={}, q={}, d={}, label:{}, relType:{})"
          .format(p, q, d, label, relType))
    embeddings = []
    with driver.session() as session:
        results = session.run(NODE2VEC_CYPHER, L=label, R=relType,
                              p=float(p), q=float(q), d=int(d))
        for result in results:
            embeddings.append(result)
    print("...generated {} embeddings".format(len(embeddings)))
    return embeddings


def kmeans(embeddings, k=NUM_CLUSTERS, clusterParam="clusterId"):
    """
    Given a list of dicts like {"nodeId" 1, "embedding": [1.0, 0.1, ...]},
    generate a list of dicts like {"nodeId": 1, "valueMap": {"clusterId": 2}}
    """
    print("Performing K-Means clustering (n_clusters={}, clusterParam={})"
          .format(NUM_CLUSTERS, clusterParam))
    X = np.array([e["embedding"] for e in embeddings])
    kmeans = KMeans(n_clusters=int(k)).fit(X)
    results = []
    for idx, cluster in enumerate(kmeans.predict(X)):
        results.append({ "nodeId": embeddings[idx]["nodeId"],
                         "valueMap": { clusterParam: int(cluster) }})
    print("...clustering completed.")
    return results


def _update_tx(tx, cypher, **kwargs):
    result = tx.run(cypher, kwargs)
    return result.consume()

def update_clusters(driver, clusterResults):
    """
    Given a list of dicts with "nodeId" string and a "valueMap" dict, update
    the graph by setting the properties from the "valueMap" on each node.
    """
    print("Updating graph...")
    with driver.session() as session:
        result = session.write_transaction(_update_tx, UPDATE_CYPHER, updates=clusterResults)
        print("...update complete: {}".format(result.counters))


if __name__ == '__main__':
    # getopt, because: "POSIX getopt(1) is The Correct Way" ~sircmpwn
    try:
        opts, args = getopt.getopt(sys.argv[1:], "hA:U:P:C:R:L:p:q:d:k:")
    except getopt.GetoptError as err:
        print(err)
        usage()

    uri = DEFAULT_URI
    user = DEFAULT_USER
    password = DEFAULT_PASS
    relType = DEFAULT_REL
    label = DEFAULT_LABEL
    clusterParam = "clusterId"
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
        elif o == "-R":
            relType = a
        elif o == "-L":
            label = a
        elif o == "-C":
            clusterParam = a
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
    embeddings = extract_embeddings(driver, label=label, relType=relType,
                                    p=p, q=q, d=d)
    clusters = kmeans(embeddings, k=k, clusterParam=clusterParam)
    update_clusters(driver, clusters)
    driver.close()
