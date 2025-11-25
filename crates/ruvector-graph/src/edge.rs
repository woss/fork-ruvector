//! Edge (relationship) implementation

use crate::types::{EdgeId, NodeId, Properties};
use bincode::{Encode, Decode};
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize, Encode, Decode)]
pub struct Edge {
    pub id: EdgeId,
    pub from: NodeId,
    pub to: NodeId,
    pub edge_type: String,
    pub properties: Properties,
}

impl Edge {
    pub fn new(
        id: EdgeId,
        from: NodeId,
        to: NodeId,
        edge_type: String,
        properties: Properties,
    ) -> Self {
        Self { id, from, to, edge_type, properties }
    }
}
