//! Vector-Graph Hybrid Query System
//!
//! Combines vector similarity search with graph traversal for AI workloads.
//! Supports semantic search, RAG (Retrieval Augmented Generation), and GNN inference.

pub mod vector_index;
pub mod semantic_search;
pub mod rag_integration;
pub mod cypher_extensions;
pub mod graph_neural;

// Re-export main types
pub use vector_index::{HybridIndex, VectorIndexType, EmbeddingConfig};
pub use semantic_search::{
    SemanticSearch, SemanticSearchConfig, SemanticPath, ClusterResult,
};
pub use rag_integration::{
    RagEngine, RagConfig, Context, ReasoningPath, Evidence,
};
pub use cypher_extensions::{
    VectorCypherParser, VectorCypherExecutor, SimilarityPredicate,
};
pub use graph_neural::{
    GraphNeuralEngine, GnnConfig, NodeClassification, LinkPrediction, GraphEmbedding,
};

use crate::error::Result;
use serde::{Deserialize, Serialize};

/// Hybrid query combining graph patterns and vector similarity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HybridQuery {
    /// Cypher pattern to match graph structure
    pub graph_pattern: String,
    /// Vector similarity constraint
    pub vector_constraint: Option<VectorConstraint>,
    /// Maximum results to return
    pub limit: usize,
    /// Minimum similarity score threshold
    pub min_score: f32,
}

/// Vector similarity constraint for hybrid queries
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorConstraint {
    /// Query embedding vector
    pub query_vector: Vec<f32>,
    /// Property name containing the embedding
    pub embedding_property: String,
    /// Top-k similar items
    pub top_k: usize,
}

/// Result from a hybrid query
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HybridResult {
    /// Matched graph elements
    pub graph_match: serde_json::Value,
    /// Similarity score
    pub score: f32,
    /// Explanation of match
    pub explanation: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hybrid_query_creation() {
        let query = HybridQuery {
            graph_pattern: "MATCH (n:Document) RETURN n".to_string(),
            vector_constraint: None,
            limit: 10,
            min_score: 0.8,
        };
        assert_eq!(query.limit, 10);
    }
}
