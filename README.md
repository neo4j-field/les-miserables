# node2vec -- reproducing Grover & Leskovec

As I work through understanding "node2vec: Scalable Feature Learning for Networks"[1], I'm experimenting with reproducing some of Grover & Leskovec's findings.

## Les Mis' Case Study
G&L use the Les Mis' data (see my [json version](./lesmis.json)) to demonstrate both _homophily_ and _structural equivalence_. The dataset has 77 Character nodes and 254 APPEARED_WITH relationships connecting them.

### Tooling
See my python code in [kmeans.py](./kmeans.py). To use it:

1. Load the [sample data](./lesmis.json) using APOC's `apoc.import.json` proc. It should just load.
2. Create a Python virtualenv and install deps: `pip install -r requirements.txt`
3. Run the code: `$ python -A bolt://localhost:7687 -U neo4j -P password -d 16 -p 1 -q 0.6`

### Les Mis Homophily
G&L claim they set `d = 16` and `p = 1, q = 0.5`.

I'm struggling to reproduce this...

...more details coming.

[1]: https://arxiv.org/pdf/1607.00653.pdf
