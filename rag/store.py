import lancedb
import os
from pathlib import Path
from rag.config import LANCE_DB_PATH

class LanceDBStore:
    def __init__(self, db_path=LANCE_DB_PATH, table_name="chunks"):
        self.db_path = db_path
        self.table_name = table_name
        self.db = lancedb.connect(self.db_path)
        self.table = None
        self._init_table()

    def _table_exists(self, table_name):
        try:
            res = self.db.list_tables()
            if hasattr(res, "tables"):
                return table_name in res.tables
            return table_name in res
        except Exception:
            return table_name in self.db.table_names()

    def _init_table(self):
        """Opens or creates the table."""
        if self._table_exists(self.table_name):
            self.table = self.db.open_table(self.table_name)
        else:
            self.table = None

    def add_chunks(self, chunks):
        """
        Adds chunks to the store.
        Each chunk is a dict:
        {
            "id": str (hash),
            "vector": list[float],
            "text": str,
            "file_path": str,
            "file_name": str,
            "chunk_index": int,
            "file_type": str,
            "created_at": float
        }
        Implements idempotency: checks if chunk ID already exists, or overwrites.
        """
        if not chunks:
            return

        # If table does not exist, create it with the first batch of chunks
        if self.table is None:
            self.table = self.db.create_table(self.table_name, data=chunks)
            return

        # Table exists. To avoid duplicate vectors, let's check existing IDs in the table
        # We can query all IDs from the table to do in-memory deduplication,
        # or we can use LanceDB merge_insert if supported.
        # Let's do a robust merge_insert or manual deduplication.
        try:
            # Get list of existing IDs
            # Since LanceDB tables can be queried using SQL/duckdb or search, let's fetch existing IDs.
            # To be efficient and generic:
            existing_df = self.table.to_pandas()
            existing_ids = set(existing_df["id"].tolist()) if "id" in existing_df.columns else set()
        except Exception as e:
            # Fallback if table is empty or has issues
            existing_ids = set()

        new_chunks = [c for c in chunks if c["id"] not in existing_ids]

        if new_chunks:
            self.table.add(new_chunks)
        
        # Refresh table reference
        self._init_table()

    def delete_by_file_path(self, file_path):
        """Deletes all chunks associated with a specific file path."""
        if self.table is not None:
            # Escape single quotes in file_path
            escaped_path = file_path.replace("'", "''")
            self.table.delete(f"file_path = '{escaped_path}'")
            self._init_table()

    def search(self, query_vector, limit=4, filter_expr=None):
        """
        Performs vector search with optional SQL-like metadata filtering.
        Returns a list of search results.
        """
        if self.table is None:
            return []
        
        query = self.table.search(query_vector)
        if filter_expr:
            query = query.where(filter_expr)
        
        results = query.limit(limit).to_pandas()
        
        # Convert pandas dataframe to list of dicts
        records = []
        for _, row in results.iterrows():
            record = row.to_dict()
            # Rename _distance to distance for consistency
            if "_distance" in record:
                record["distance"] = record.pop("_distance")
            records.append(record)
        return records

    def count_vectors(self):
        """Returns the number of vectors in the table."""
        if self.table is None:
            return 0
        return len(self.table)

    def clear_all(self):
        """Deletes the table and recreates it empty."""
        if self._table_exists(self.table_name):
            self.db.drop_table(self.table_name)
        self.table = None
