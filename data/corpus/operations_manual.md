# AetherDB Operations Manual

## Cluster Requirements & Scaling
To ensure stable production operations, AetherDB nodes must meet the following hardware criteria:
- **Minimum RAM**: **8 GB** per node (16 GB recommended for production clusters).
- **CPU**: **4 Cores** minimum.
- **Disk**: High-performance NVMe SSDs are highly recommended to prevent latency spikes during MemTable flushes and SSTable compactions.

## Scaling Out
To add a node to the cluster:
1. Provision a host matching cluster requirements.
2. Edit `aether.conf` and set `seed_nodes` to point to the active seeds of the cluster.
3. Start the node service. It will connect to seed nodes via port **7070** (Gossip protocol) and begin bootstrap syncing.

## Backing Up Data
Backup operations are fully online and do not block writes:
- Run <code>aether-admin backup create --destination /mnt/backup</code>
- AetherDB will take a snapshot of the SSTables in the storage directory and copy them to the backup directory.
- Transaction logs (WAL) are flushed to disk before the snapshot is executed.
- Backup frequency is recommended to be daily.

## Troubleshooting Port Conflicts
If port **8080** is already in use by another service on the host, you can edit the configuration file to change the <code>http_port</code> parameter. 
Ensure ports **9090** (Raft) and **7070** (Gossip) are also open in your firewall rules for node-to-node communication.
