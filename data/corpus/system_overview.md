# AetherDB: High-Performance Distributed Storage System

## System Overview
AetherDB is a next-generation distributed, partition-tolerant, and highly consistent NoSQL database designed for low-latency write-heavy workloads. 
It uses a decentralized masterless-like coordinate topology with strong consistency guarantees using a customized consensus model.

## Architecture & Storage Engine
Under the hood, AetherDB relies on a log-structured merge-tree (LSM-tree) storage engine for disk writes. 
Incoming writes are first appended to a commit log (Write-Ahead Log or WAL) and then written to an in-memory MemTable. 
When the MemTable reaches 64MB in size, it is flushed to disk as an immutable SSTable (Sorted String Table).

## Replication & Consensus
- **Default Replication Factor**: The default replication factor for AetherDB is **3**, meaning every partition is replicated across three separate nodes in the cluster.
- **Consensus Protocol**: AetherDB utilizes the **Raft** consensus protocol to manage leader election and ensure strong consistency across write operations.
- **Port Mapping**:
  - Client REST API Port: **8080**
  - Raft Inter-Node Consensus Port: **9090**
  - Gossip Membership Port: **7070**

## Technical Limits
- Maximum key size: **2 KB**
- Maximum value size: **50 MB**
- Maximum active connections per node: **10,000**
