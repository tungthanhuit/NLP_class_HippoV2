#!/usr/bin/env python3

import argparse
import os
import sys
from pathlib import Path
from typing import Optional


def _maybe_load_dotenv(dotenv_path: Optional[str]) -> None:
    try:
        from dotenv import load_dotenv  # type: ignore

        if dotenv_path:
            load_dotenv(dotenv_path, override=False)
        else:
            # Default behavior: load .env in CWD if present.
            load_dotenv(override=False)
    except Exception:
        # Best-effort; do not hard-require python-dotenv.
        return


def _labelize(model_name: str) -> str:
    return (model_name or "").replace("/", "_")


def _compute_effective_namespace(
    neo4j_namespace: str,
    llm_name: str,
    embedding_name: str,
    include_llm: bool,
) -> str:
    prefix = (neo4j_namespace or "").strip()
    if not prefix:
        raise ValueError("neo4j_namespace must be non-empty")

    llm_label = _labelize(llm_name)
    emb_label = _labelize(embedding_name)
    if not emb_label:
        raise ValueError("embedding_name must be non-empty")

    parts = [prefix]
    if include_llm:
        if not llm_label:
            raise ValueError("llm_name must be non-empty when include_llm is true")
        parts.append(llm_label)
    parts.append(emb_label)
    return "_".join(parts)


def _drop_milvus_collection(
    *,
    milvus_uri: str,
    milvus_token: Optional[str],
    collection_name: str,
) -> bool:
    # Import lazily to keep Neo4j-only usage lightweight.
    from pymilvus import connections, utility  # type: ignore

    # Keep a dedicated alias per process.
    alias = f"hipporag_{collection_name}"

    try:
        if milvus_token:
            connections.connect(alias=alias, uri=milvus_uri, token=milvus_token)
        else:
            connections.connect(alias=alias, uri=milvus_uri)
    except TypeError:
        # Older pymilvus may not support `token=`.
        connections.connect(alias=alias, uri=milvus_uri)

    if not utility.has_collection(collection_name, using=alias):
        return False

    utility.drop_collection(collection_name, using=alias)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Delete all HippoRAG data for a Neo4j namespace (nodes + relationships + embeddings). "
            "Optionally drop the matching Milvus chunk-vector collection."
        )
    )

    parser.add_argument(
        "--dotenv",
        type=str,
        default=None,
        help="Optional path to a .env file to load (best-effort; does not override existing env vars).",
    )

    # Neo4j
    parser.add_argument("--neo4j-uri", type=str, default=os.getenv("NEO4J_URI", "bolt://localhost:7687"))
    parser.add_argument("--neo4j-user", type=str, default=os.getenv("NEO4J_USER", "neo4j"))
    parser.add_argument("--neo4j-password", type=str, default=os.getenv("NEO4J_PASSWORD"))
    parser.add_argument("--neo4j-database", type=str, default=os.getenv("NEO4J_DATABASE", "neo4j"))

    parser.add_argument(
        "--list-namespaces",
        action="store_true",
        help="List distinct Neo4j `ns` values and counts in the selected database, then exit.",
    )

    # Namespace selection
    parser.add_argument(
        "--namespace",
        type=str,
        default=None,
        help=(
            "Effective Neo4j namespace to clear (this is the exact string stored in n.ns). "
            "If provided, it takes precedence over the computed namespace flags below."
        ),
    )
    parser.add_argument(
        "--neo4j-namespace",
        type=str,
        default=os.getenv("NEO4J_NAMESPACE", "hipporag"),
        help="Neo4j namespace prefix used by HippoRAG (BaseConfig.neo4j_namespace).",
    )
    parser.add_argument(
        "--llm-name",
        type=str,
        default=os.getenv("HIPPORAG_LLM_NAME") or os.getenv("LLM_NAME") or os.getenv("OPENAI_MODEL"),
        help="LLM model name used by HippoRAG (BaseConfig.llm_name). Only needed to compute the effective namespace.",
    )
    parser.add_argument(
        "--embedding-name",
        type=str,
        default=os.getenv("HIPPORAG_EMBEDDING_MODEL_NAME") or os.getenv("EMBEDDING_MODEL_NAME"),
        help="Embedding model name used by HippoRAG (BaseConfig.embedding_model_name). Only needed to compute the effective namespace.",
    )
    parser.add_argument(
        "--include-llm",
        type=str,
        default=os.getenv("HIPPORAG_NEO4J_NAMESPACE_INCLUDE_LLM", "true"),
        help="Whether the effective namespace includes the llm label (BaseConfig.neo4j_namespace_include_llm).",
    )

    # Milvus (optional)
    parser.add_argument(
        "--drop-milvus",
        type=str,
        default=os.getenv("HIPPORAG_DROP_MILVUS", "false"),
        help="If true, drop the Milvus chunk-vector collection that corresponds to the effective namespace.",
    )
    parser.add_argument(
        "--milvus-uri",
        type=str,
        default=os.getenv("MILVUS_URI"),
        help="Milvus URI (required if --drop-milvus true).",
    )
    parser.add_argument(
        "--milvus-token",
        type=str,
        default=os.getenv("MILVUS_TOKEN"),
        help="Milvus token (optional).",
    )
    parser.add_argument(
        "--milvus-collection-prefix",
        type=str,
        default=os.getenv("HIPPORAG_MILVUS_COLLECTION_PREFIX", "hipporag"),
        help="Milvus collection prefix (BaseConfig.milvus_collection_prefix).",
    )

    # Safety
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Actually perform deletion. Without this flag, the script prints the plan and exits.",
    )

    args = parser.parse_args()

    # Make `src.*` imports work even when running from outside repo root.
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    _maybe_load_dotenv(args.dotenv)

    if not args.neo4j_password:
        raise SystemExit("NEO4J_PASSWORD is required (set env or pass --neo4j-password)")

    def _as_bool(v: str) -> bool:
        return str(v).strip().lower() in {"1", "true", "yes", "y", "t"}

    include_llm = _as_bool(args.include_llm)
    drop_milvus = _as_bool(args.drop_milvus)

    if args.list_namespaces:
        try:
            from neo4j import GraphDatabase  # type: ignore

            driver = GraphDatabase.driver(
                args.neo4j_uri,
                auth=(args.neo4j_user, args.neo4j_password),
            )
            try:
                query = (
                    "MATCH (n) WHERE n.ns IS NOT NULL "
                    "RETURN n.ns AS ns, count(n) AS cnt "
                    "ORDER BY cnt DESC"
                )
                with driver.session(database=args.neo4j_database) as session:
                    rows = session.execute_read(lambda tx: list(tx.run(query)))

                if not rows:
                    print("No nodes with `ns` found in this database.")
                    return 0

                print("Namespaces in Neo4j:")
                for r in rows:
                    print(f"  {r['ns']}: {int(r['cnt'])}")
                return 0
            finally:
                try:
                    driver.close()
                except Exception:
                    pass
        except Exception as e:
            raise SystemExit(f"Failed listing namespaces: {e}")

    user_ns_arg = (args.namespace or "").strip()

    effective_ns = user_ns_arg
    if not effective_ns:
        effective_ns = _compute_effective_namespace(
            neo4j_namespace=args.neo4j_namespace,
            llm_name=args.llm_name or "",
            embedding_name=args.embedding_name or "",
            include_llm=include_llm,
        )

    if drop_milvus:
        if not args.milvus_uri:
            raise SystemExit("--milvus-uri (or env MILVUS_URI) is required when --drop-milvus is true")

        # Compute collection name exactly like HippoRAG.
        raw_collection = f"{args.milvus_collection_prefix}_{effective_ns}_chunk"
        try:
            from src.hipporag.kb.milvus_store import _sanitize_collection_name  # type: ignore

            milvus_collection = _sanitize_collection_name(raw_collection)
        except Exception:
            milvus_collection = raw_collection

    else:
        milvus_collection = None

    print("Planned deletion:")
    print(f"  Neo4j URI:      {args.neo4j_uri}")
    print(f"  Neo4j DB:       {args.neo4j_database}")
    print(f"  Neo4j ns:       {effective_ns}")
    if milvus_collection:
        print(f"  Milvus URI:     {args.milvus_uri}")
        print(f"  Milvus col:     {milvus_collection}")

    if not args.yes:
        print("\nDry-run only. Re-run with --yes to execute.")
        return 2

    try:
        from src.hipporag.kb.neo4j_store import clear_neo4j_namespace as _clear_neo4j_namespace
    except Exception:
        # Fallback for local edits/older versions.
        from src.hipporag.kb.neo4j_store import Neo4jConfig, Neo4jKB

        def _clear_neo4j_namespace(
            *,
            uri: str,
            user: str,
            password: str,
            namespace: str,
            database: str = "neo4j",
        ) -> int:
            kb = Neo4jKB(
                Neo4jConfig(
                    uri=uri,
                    user=user,
                    password=password,
                    database=database,
                    namespace=namespace,
                )
            )
            try:
                return kb.clear_namespace(namespace=namespace, database=database)
            finally:
                kb.close()

    deleted_nodes = _clear_neo4j_namespace(
        uri=args.neo4j_uri,
        user=args.neo4j_user,
        password=args.neo4j_password,
        namespace=effective_ns,
        database=args.neo4j_database,
    )

    print(f"Deleted Neo4j nodes (best-effort count): {deleted_nodes}")

    if int(deleted_nodes) == 0:
        print(
            "Hint: HippoRAG usually stores `ns` as '<neo4j_namespace>_<llm>_<embedding>' "
            "(or '<neo4j_namespace>_<embedding>' when include_llm is false). "
            "Try running with computed namespace flags (omit --namespace), or run with --list-namespaces to discover the exact `ns` values."
        )

        # If the user passed a short prefix like "hipporag", show candidates.
        if user_ns_arg:
            try:
                from neo4j import GraphDatabase  # type: ignore

                driver = GraphDatabase.driver(
                    args.neo4j_uri,
                    auth=(args.neo4j_user, args.neo4j_password),
                )
                try:
                    q = (
                        "MATCH (n) "
                        "WHERE n.ns IS NOT NULL AND n.ns STARTS WITH $prefix "
                        "RETURN n.ns AS ns, count(n) AS cnt "
                        "ORDER BY cnt DESC LIMIT 20"
                    )
                    with driver.session(database=args.neo4j_database) as session:
                        rows = session.execute_read(
                            lambda tx: list(tx.run(q, prefix=user_ns_arg))
                        )
                    if rows:
                        print("Namespace candidates (starts with your --namespace):")
                        for r in rows:
                            print(f"  {r['ns']}: {int(r['cnt'])}")
                finally:
                    try:
                        driver.close()
                    except Exception:
                        pass
            except Exception:
                # Best-effort only.
                pass

    if milvus_collection:
        dropped = _drop_milvus_collection(
            milvus_uri=args.milvus_uri,
            milvus_token=args.milvus_token,
            collection_name=milvus_collection,
        )
        print(f"Milvus collection dropped: {bool(dropped)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
