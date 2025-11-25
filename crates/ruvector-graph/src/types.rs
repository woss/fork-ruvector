//! Core types for graph database

use bincode::{Encode, Decode};
use serde::{Serialize, Deserialize};
use std::collections::HashMap;

pub type NodeId = String;
pub type EdgeId = String;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Encode, Decode)]
pub enum PropertyValue {
    String(String),
    Integer(i64),
    Float(f64),
    Boolean(bool),
    List(Vec<PropertyValue>),
    Null,
}

pub type Properties = HashMap<String, PropertyValue>;

#[derive(Debug, Clone, Serialize, Deserialize, Encode, Decode)]
pub struct Label {
    pub name: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Encode, Decode)]
pub struct RelationType {
    pub name: String,
}
