#!/usr/bin/env python
import getopt, sys
from neo4j import GraphDatabase
from sklearn.cluster import KMeans
import numpy as np

# Global defaults, some based on our demo and some on the algo defaults.
DEFAULT_URI = "bolt://localhost:7687"
DEFAULT_USER = "neo4j"
DEFAULT_PASS = "password"
DEFAULT_REL = "UNWEIGHTED_APPEARED_WITH"
DEFAULT_LABEL = "Character"
DEFAULT_PROP = "clusterId"
DEFAULT_P = 1.0
DEFAULT_Q = 1.0
DEFAULT_D = 16
DEFAULT_WALK = 80
DEFAULT_K=6

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
  inOutFactor: $q,
  walkLength: $l
}) YIELD nodeId, embedding
"""

UPDATE_CYPHER = """
UNWIND $updates AS updateMap
    MATCH (n) WHERE id(n) = updateMap.nodeId
    SET n += updateMap.valueMap
"""

def usage():
    print("usage:\t kmeans.py [-A BOLT URI default: {}] [-U USERNAME (default: {})] [-P PASSWORD (default: {})]"
          .format(DEFAULT_URI, DEFAULT_USER, DEFAULT_PASS))
    print("supported parameters:")
    print("\t-R RELATIONSHIP_TYPE (default: '{}'".format(DEFAULT_REL))
    print("\t-L NODE_LABEL (default: '{}'".format(DEFAULT_LABEL))
    print("\t-C CLUSTER_PROPERTY (default: '{}'".format(DEFAULT_PROP))
    print("\t-d DIMENSIONS (default: {})".format(DEFAULT_D))
    print("\t-p RETURN PARAMETER (default: {})".format(DEFAULT_P))
    print("\t-q IN-OUT PARAMETER (default: {})".format(DEFAULT_Q))
    print("\t-k K-MEANS NUM_CLUSTERS (default: {})".format(DEFAULT_K))
    print("\t-l WALK_LENGTH (default: {})".format(DEFAULT_WALK))
    sys.exit(1)

def extract_embeddings(driver, label=DEFAULT_LABEL, relType=DEFAULT_REL,
                       p=DEFAULT_P, q=DEFAULT_Q, d=DEFAULT_D, l=DEFAULT_WALK):
    """
    Call the GDS neo2vec routine using the given driver and provided params.
    """
    print("Generating graph embeddings (p={}, q={}, d={}, l={}, label={}, relType={})"
          .format(p, q, d, l, label, relType))
    embeddings = []
    with driver.session() as session:
        results = session.run(NODE2VEC_CYPHER, L=label, R=relType,
                              p=float(p), q=float(q), d=int(d), l=int(l))
        for result in results:
            embeddings.append(result)
    print("...generated {} embeddings".format(len(embeddings)))
    return embeddings


def kmeans(embeddings, k=DEFAULT_K, clusterProp=DEFAULT_PROP):
    """
    Given a list of dicts like {"nodeId" 1, "embedding": [1.0, 0.1, ...]},
    generate a list of dicts like {"nodeId": 1, "valueMap": {"clusterId": 2}}
    """
    print("Performing K-Means clustering (k={}, clusterProp='{}')"
          .format(k, clusterProp))
    X = np.array([e["embedding"] for e in embeddings])
    kmeans = KMeans(n_clusters=int(k)).fit(X)
    results = []
    for idx, cluster in enumerate(kmeans.predict(X)):
        results.append({ "nodeId": embeddings[idx]["nodeId"],
                         "valueMap": { clusterProp: int(cluster) }})
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
        opts, args = getopt.getopt(sys.argv[1:], "hA:U:P:C:R:l:L:p:q:d:k:")
    except getopt.GetoptError as err:
        print(err)
        usage()

    uri = DEFAULT_URI
    user = DEFAULT_USER
    password = DEFAULT_PASS
    relType = DEFAULT_REL
    label = DEFAULT_LABEL
    clusterProp = DEFAULT_PROP
    p = DEFAULT_P
    q = DEFAULT_Q
    d = DEFAULT_D
    k = DEFAULT_K
    l = DEFAULT_WALK

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
            clusterProp = a
        elif o == "-p":
            p = a
        elif o == "-q":
            q = a
        elif o == "-d":
            d = a
        elif o == "-k":
            k = a
        elif o == "-l":
            l = a
        else:
            usage()

    print("Connecting to uri: {}".format(uri))
    driver = GraphDatabase.driver(uri, auth=(user, password))
    embeddings = extract_embeddings(driver, label=label, relType=relType,
                                    p=p, q=q, d=d, l=l)
    clusters = kmeans(embeddings, k=k, clusterProp=clusterProp)
    update_clusters(driver, clusters)
    driver.close()
