//! Node implementation

use crate::types::{NodeId, Properties, Label};
use bincode::{Encode, Decode};
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize, Encode, Decode)]
pub struct Node {
    pub id: NodeId,
    pub labels: Vec<Label>,
    pub properties: Properties,
}

impl Node {
    pub fn new(id: NodeId, labels: Vec<Label>, properties: Properties) -> Self {
        Self { id, labels, properties }
    }
}
