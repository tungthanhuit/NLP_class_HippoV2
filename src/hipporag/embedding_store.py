import numpy as np
import os
from typing import List
import logging
from copy import deepcopy
import pandas as pd
import time

from .utils.misc_utils import compute_mdhash_id

logger = logging.getLogger(__name__)


class EmbeddingStore:
    def __init__(self, embedding_model, db_filename, batch_size, namespace):
        """
        Initializes the class with necessary configurations and sets up the working directory.

        Parameters:
        embedding_model: The model used for embeddings.
        db_filename: The directory path where data will be stored or retrieved.
        batch_size: The batch size used for processing.
        namespace: A unique identifier for data segregation.

        Functionality:
        - Assigns the provided parameters to instance variables.
        - Checks if the directory specified by `db_filename` exists.
          - If not, creates the directory and logs the operation.
        - Constructs the filename for storing data in a parquet file format.
        - Calls the method `_load_data()` to initialize the data loading process.
        """
        self.embedding_model = embedding_model
        self.batch_size = batch_size
        self.namespace = namespace

        if not os.path.exists(db_filename):
            logger.info(f"Creating working directory: {db_filename}")
            os.makedirs(db_filename, exist_ok=True)

        self.filename = os.path.join(db_filename, f"vdb_{self.namespace}.parquet")
        self._load_data()

    def get_missing_string_hash_ids(self, texts: List[str]):
        nodes_dict = {}

        for text in texts:
            nodes_dict[compute_mdhash_id(text, prefix=self.namespace + "-")] = {
                "content": text
            }

        # Get all hash_ids from the input dictionary.
        all_hash_ids = list(nodes_dict.keys())
        if not all_hash_ids:
            return {}

        existing = self.hash_id_to_row.keys()

        # Filter out the missing hash_ids.
        missing_ids = [hash_id for hash_id in all_hash_ids if hash_id not in existing]
        texts_to_encode = [nodes_dict[hash_id]["content"] for hash_id in missing_ids]

        return {
            h: {"hash_id": h, "content": t}
            for h, t in zip(missing_ids, texts_to_encode)
        }

    def insert_strings(self, texts: List[str]):
        return self.insert_strings_with_embedding_texts(texts, texts)

    def insert_strings_with_embedding_texts(
        self, texts: List[str], embedding_texts: List[str]
    ):
        if len(texts) != len(embedding_texts):
            raise ValueError(
                f"texts and embedding_texts must have the same length, got {len(texts)} and {len(embedding_texts)}"
            )

        nodes_dict = {}

        for text, embedding_text in zip(texts, embedding_texts):
            nodes_dict[compute_mdhash_id(text, prefix=self.namespace + "-")] = {
                "content": text,
                "embedding_text": embedding_text,
            }

        # Get all hash_ids from the input dictionary.
        all_hash_ids = list(nodes_dict.keys())
        if not all_hash_ids:
            return  # Nothing to insert.

        existing = self.hash_id_to_row.keys()

        # Filter out the missing hash_ids.
        missing_ids = [hash_id for hash_id in all_hash_ids if hash_id not in existing]
        stale_ids = [
            hash_id
            for hash_id in all_hash_ids
            if hash_id in existing
            and self.hash_id_to_row[hash_id].get(
                "embedding_text", self.hash_id_to_row[hash_id]["content"]
            )
            != nodes_dict[hash_id]["embedding_text"]
        ]

        logger.info(
            f"Inserting {len(missing_ids)} new records, refreshing {len(stale_ids)} records, "
            f"{len(all_hash_ids) - len(missing_ids) - len(stale_ids)} records already current."
        )

        if len(missing_ids) == 0 and len(stale_ids) == 0:
            # Helpful console signal that embeddings are coming from cache.
            print(
                f"[EmbeddingStore:{self.namespace}] cache hit: {len(all_hash_ids)} records already exist"
            )

        ids_to_encode = missing_ids + stale_ids

        if not ids_to_encode:
            return {}  # All records already exist.

        # Encode from the retrieval surface while preserving the canonical content.
        texts_to_encode = [
            nodes_dict[hash_id]["embedding_text"] for hash_id in ids_to_encode
        ]

        print(
            f"[EmbeddingStore:{self.namespace}] encoding {len(texts_to_encode)} embedding texts..."
        )
        start_time = time.time()
        missing_embeddings = self.embedding_model.batch_encode(texts_to_encode)
        elapsed = time.time() - start_time

        try:
            emb_shape = np.asarray(missing_embeddings).shape
        except Exception:
            emb_shape = "(unknown)"
        print(
            f"[EmbeddingStore:{self.namespace}] encoded shape={emb_shape} in {elapsed:.2f}s"
        )

        self._upsert(
            ids_to_encode,
            [nodes_dict[hash_id]["content"] for hash_id in ids_to_encode],
            missing_embeddings,
            [nodes_dict[hash_id]["embedding_text"] for hash_id in ids_to_encode],
        )

    def _load_data(self):
        if os.path.exists(self.filename):
            df = pd.read_parquet(self.filename)
            self.hash_ids, self.texts, self.embeddings = (
                df["hash_id"].values.tolist(),
                df["content"].values.tolist(),
                df["embedding"].values.tolist(),
            )
            self.embedding_texts = (
                df["embedding_text"].values.tolist()
                if "embedding_text" in df.columns
                else self.texts.copy()
            )
            self.hash_id_to_idx = {h: idx for idx, h in enumerate(self.hash_ids)}
            self.hash_id_to_row = {
                h: {"hash_id": h, "content": t, "embedding_text": et}
                for h, t, et in zip(self.hash_ids, self.texts, self.embedding_texts)
            }
            self.hash_id_to_text = {
                h: self.texts[idx] for idx, h in enumerate(self.hash_ids)
            }
            self.text_to_hash_id = {
                self.texts[idx]: h for idx, h in enumerate(self.hash_ids)
            }
            assert (
                len(self.hash_ids)
                == len(self.texts)
                == len(self.embeddings)
                == len(self.embedding_texts)
            )
            logger.info(f"Loaded {len(self.hash_ids)} records from {self.filename}")
        else:
            self.hash_ids, self.texts, self.embeddings, self.embedding_texts = (
                [],
                [],
                [],
                [],
            )
            self.hash_id_to_idx, self.hash_id_to_row = {}, {}

    def _save_data(self):
        data_to_save = pd.DataFrame(
            {
                "hash_id": self.hash_ids,
                "content": self.texts,
                "embedding_text": self.embedding_texts,
                "embedding": self.embeddings,
            }
        )
        data_to_save.to_parquet(self.filename, index=False)
        self.hash_id_to_row = {
            h: {"hash_id": h, "content": t, "embedding_text": et}
            for h, t, et in zip(self.hash_ids, self.texts, self.embedding_texts)
        }
        self.hash_id_to_idx = {h: idx for idx, h in enumerate(self.hash_ids)}
        self.hash_id_to_text = {
            h: self.texts[idx] for idx, h in enumerate(self.hash_ids)
        }
        self.text_to_hash_id = {
            self.texts[idx]: h for idx, h in enumerate(self.hash_ids)
        }
        logger.info(f"Saved {len(self.hash_ids)} records to {self.filename}")

    def _upsert(self, hash_ids, texts, embeddings, embedding_texts=None):
        if embedding_texts is None:
            embedding_texts = texts

        for hash_id, text, embedding, embedding_text in zip(
            hash_ids, texts, embeddings, embedding_texts
        ):
            if hash_id in self.hash_id_to_idx:
                idx = self.hash_id_to_idx[hash_id]
                self.texts[idx] = text
                self.embeddings[idx] = embedding
                self.embedding_texts[idx] = embedding_text
            else:
                self.hash_ids.append(hash_id)
                self.texts.append(text)
                self.embeddings.append(embedding)
                self.embedding_texts.append(embedding_text)

        logger.info(f"Saving new records.")
        self._save_data()

    def delete(self, hash_ids):
        indices = []

        for hash in hash_ids:
            indices.append(self.hash_id_to_idx[hash])

        sorted_indices = np.sort(indices)[::-1]

        for idx in sorted_indices:
            self.hash_ids.pop(idx)
            self.texts.pop(idx)
            self.embeddings.pop(idx)
            self.embedding_texts.pop(idx)

        logger.info(f"Saving record after deletion.")
        self._save_data()

    def get_row(self, hash_id):
        return self.hash_id_to_row[hash_id]

    def get_hash_id(self, text):
        return self.text_to_hash_id[text]

    def get_rows(self, hash_ids, dtype=np.float32):
        if not hash_ids:
            return {}

        results = {id: self.hash_id_to_row[id] for id in hash_ids}

        return results

    def get_all_ids(self):
        return deepcopy(self.hash_ids)

    def get_all_id_to_rows(self):
        return deepcopy(self.hash_id_to_row)

    def get_all_texts(self):
        return set(row["content"] for row in self.hash_id_to_row.values())

    def get_embedding(self, hash_id, dtype=np.float32) -> np.ndarray:
        return self.embeddings[self.hash_id_to_idx[hash_id]].astype(dtype)

    def get_embeddings(self, hash_ids, dtype=np.float32) -> list[np.ndarray]:
        if not hash_ids:
            return []

        indices = np.array([self.hash_id_to_idx[h] for h in hash_ids], dtype=np.intp)
        embeddings = np.array(self.embeddings, dtype=dtype)[indices]

        return embeddings
