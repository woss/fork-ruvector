//! Performance optimization modules for orders of magnitude speedup
//!
//! This module provides cutting-edge optimizations targeting 100x performance
//! improvement over Neo4j through:
//! - SIMD-vectorized graph traversal
//! - Cache-optimized data layouts
//! - Custom memory allocators
//! - Compressed indexes
//! - JIT-compiled query operators
//! - Bloom filters for negative lookups
//! - Adaptive radix trees for property indexes

pub mod simd_traversal;
pub mod cache_hierarchy;
pub mod memory_pool;
pub mod index_compression;
pub mod query_jit;
pub mod bloom_filter;
pub mod adaptive_radix;

// Re-exports for convenience
pub use simd_traversal::{SimdTraversal, SimdBfsIterator, SimdDfsIterator};
pub use cache_hierarchy::{CacheHierarchy, HotColdStorage};
pub use memory_pool::{ArenaAllocator, QueryArena, NumaAllocator};
pub use index_compression::{CompressedIndex, RoaringBitmapIndex, DeltaEncoder};
pub use query_jit::{JitCompiler, JitQuery, QueryOperator};
pub use bloom_filter::{BloomFilter, ScalableBloomFilter};
pub use adaptive_radix::{AdaptiveRadixTree, ArtNode};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_optimization_modules_compile() {
        // Smoke test to ensure all modules compile
        assert!(true);
    }
}
