import json
import os
from pathlib import Path

def generate_test_suite():
    suite = [
        {
            "id": "case1",
            "query": "What is the default replication factor for AetherDB?",
            "context": "AetherDB is a distributed database. The default replication factor is 3, which writes data to three separate nodes.",
            "expected_output": "The default replication factor is 3.",
            "response_a": "AetherDB replicates all partitions to exactly 3 nodes by default, ensuring high availability.",
            "response_b": "The default replication factor for AetherDB is 3."
        },
        {
            "id": "case2",
            "query": "What consensus protocol does AetherDB use?",
            "context": "Replication and consensus are managed by Raft in AetherDB. Raft handles leader elections and log replication.",
            "expected_output": "AetherDB uses Raft consensus protocol.",
            "response_a": "AetherDB uses Raft consensus protocol for consensus and replication.",
            "response_b": "AetherDB consensus is handled by Raft protocol, which is a consensus protocol."
        },
        {
            "id": "case3",
            "query": "Which port is used for Raft Inter-Node Consensus?",
            "context": "AetherDB client port is 8080. The Raft consensus port is 9090. Gossip runs on 7070.",
            "expected_output": "Port 9090 is used for Raft inter-node consensus.",
            "response_a": "The Raft consensus protocol runs on port 9090 for communication between nodes.",
            "response_b": "AetherDB uses port 9090 for Raft consensus."
        },
        {
            "id": "case4",
            "query": "What is the maximum key size in AetherDB?",
            "context": "Technical limits of AetherDB include: maximum key size is 2 KB, value size is 50 MB.",
            "expected_output": "The maximum key size is 2 KB.",
            "response_a": "The key size limit is 2 KB in AetherDB.",
            "response_b": "The maximum key size that AetherDB can store is 2 KB, whereas values can be up to 50 MB."
        },
        {
            "id": "case5",
            "query": "What happens when the MemTable reaches 64MB?",
            "context": "Writes go to a MemTable first. When the MemTable reaches 64MB, it is flushed to disk as an SSTable.",
            "expected_output": "It is flushed to disk as an immutable SSTable.",
            "response_a": "When it reaches 64MB, AetherDB flushes the MemTable to disk as an SSTable.",
            "response_b": "It is flushed to disk as an immutable SSTable."
        },
        {
            "id": "case6",
            "query": "What is the minimum RAM required for AetherDB?",
            "context": "Node hardware requirements: Minimum RAM is 8 GB, CPU is 4 cores.",
            "expected_output": "The minimum RAM required is 8 GB.",
            "response_a": "AetherDB requires a minimum of 8 GB RAM per node, but 16 GB is recommended.",
            "response_b": "The minimum RAM required is 8 GB."
        },
        {
            "id": "case7",
            "query": "What HTTP header is required for REST API requests?",
            "context": "REST API writes and reads require the X-Aether-Token header for authorization.",
            "expected_output": "The X-Aether-Token HTTP header is required.",
            "response_a": "You need to include the X-Aether-Token header in the HTTP request headers.",
            "response_b": "Requests require X-Aether-Token header."
        },
        {
            "id": "case8",
            "query": "Does AetherDB support external Key Management Services?",
            "context": "Encryption keys for AetherDB SSTables must be managed using an external KMS (Key Management Service).",
            "expected_output": "Yes, AetherDB supports managing keys using an external Key Management Service.",
            "response_a": "Yes, AetherDB supports managing encryption keys with an external KMS.",
            "response_b": "AetherDB supports KMS for key management."
        },
        {
            "id": "case9",
            "query": "What encryption algorithm is supported for data at rest?",
            "context": "For data at rest, AetherDB supports AES-256 block encryption for SSTables and WAL.",
            "expected_output": "AES-256 block encryption is supported.",
            "response_a": "AES-256 block encryption is used for data at rest.",
            "response_b": "AetherDB supports AES-256 encryption."
        },
        {
            "id": "case10",
            "query": "How do you scale out a cluster by adding a node?",
            "context": "To scale out, edit aether.conf and set seed_nodes to point to active seed nodes, then start the service.",
            "expected_output": "Edit aether.conf, set seed_nodes to the active seeds, and start the node service.",
            "response_a": "You add a node by setting the seed_nodes parameter in aether.conf and starting the service.",
            "response_b": "Set seed_nodes in aether.conf and start the node."
        },
        {
            "id": "case11",
            "query": "Are backups online or offline?",
            "context": "AetherDB backup operations are online and do not block writes.",
            "expected_output": "Backups are fully online and do not block writes.",
            "response_a": "Backups are online, meaning writes are not blocked during snapshotting.",
            "response_b": "Backups are fully online."
        },
        {
            "id": "case12",
            "query": "Where are backups stored by the backup command?",
            "context": "Run backup command with --destination flag. It copies snapshot SSTables to the destination folder.",
            "expected_output": "They are stored in the folder specified by the --destination flag.",
            "response_a": "They are stored in the directory passed to the --destination flag.",
            "response_b": "Stored in the destination directory."
        },
        {
            "id": "case13",
            "query": "What is the CPU core requirement?",
            "context": "Minimum hardware requirement: 4 CPU Cores, 8 GB RAM.",
            "expected_output": "A minimum of 4 CPU Cores is required.",
            "response_a": "AetherDB requires a minimum of 4 CPU cores per node.",
            "response_b": "The minimum requirement is 4 Cores."
        },
        {
            "id": "case14",
            "query": "Does the health check endpoint require authentication?",
            "context": "The health check endpoint /api/v1/health does not require X-Aether-Token.",
            "expected_output": "No, it does not require authentication.",
            "response_a": "No, the /api/v1/health endpoint does not require authentication headers.",
            "response_b": "No, /api/v1/health does not require token."
        },
        {
            "id": "case15",
            "query": "What is the gossip port used for?",
            "context": "Port 7070 is used for Gossip membership tracking and node discovery.",
            "expected_output": "Port 7070 is used for Gossip membership protocol and discovery.",
            "response_a": "Port 7070 is utilized by AetherDB's Gossip membership protocol.",
            "response_b": "Gossip membership runs on 7070."
        }
    ]

    out_dir = Path(__file__).resolve().parent.parent / "data"
    os.makedirs(out_dir, exist_ok=True)
    with open(out_dir / "test_suite.json", "w") as f:
        json.dump(suite, f, indent=2)
    print(f"Generated test suite with {len(suite)} cases at: {out_dir / 'test_suite.json'}")

if __name__ == "__main__":
    generate_test_suite()
