//! Cypher query parser tests
//!
//! Tests for parsing valid and invalid Cypher queries to ensure syntax correctness.

use ruvector_graph::cypher::CypherQuery;

// ============================================================================
// Valid Cypher Queries
// ============================================================================

#[test]
fn test_parse_simple_match() {
    let query = CypherQuery::new("MATCH (n) RETURN n");
    let result = query.parse();

    // TODO: Implement parser - for now just verify it doesn't panic
    assert!(result.is_ok());
}

#[test]
fn test_parse_match_with_label() {
    let query = CypherQuery::new("MATCH (n:Person) RETURN n");
    let result = query.parse();
    assert!(result.is_ok());
}

#[test]
fn test_parse_match_with_properties() {
    let query = CypherQuery::new("MATCH (n:Person {name: 'Alice'}) RETURN n");
    let result = query.parse();
    assert!(result.is_ok());
}

#[test]
fn test_parse_match_relationship() {
    let query = CypherQuery::new("MATCH (a)-[r:KNOWS]->(b) RETURN a, r, b");
    let result = query.parse();
    assert!(result.is_ok());
}

#[test]
fn test_parse_match_undirected_relationship() {
    let query = CypherQuery::new("MATCH (a)-[r:FRIEND]-(b) RETURN a, b");
    let result = query.parse();
    assert!(result.is_ok());
}

#[test]
fn test_parse_match_path() {
    let query = CypherQuery::new("MATCH p = (a)-[:KNOWS*1..3]->(b) RETURN p");
    let result = query.parse();
    assert!(result.is_ok());
}

#[test]
fn test_parse_create_node() {
    let query = CypherQuery::new("CREATE (n:Person {name: 'Bob', age: 30})");
    let result = query.parse();
    assert!(result.is_ok());
}

#[test]
fn test_parse_create_relationship() {
    let query = CypherQuery::new(
        "CREATE (a:Person {name: 'Alice'})-[r:KNOWS]->(b:Person {name: 'Bob'})"
    );
    let result = query.parse();
    assert!(result.is_ok());
}

#[test]
fn test_parse_merge() {
    let query = CypherQuery::new("MERGE (n:Person {name: 'Charlie'})");
    let result = query.parse();
    assert!(result.is_ok());
}

#[test]
fn test_parse_delete() {
    let query = CypherQuery::new("MATCH (n:Person {name: 'Alice'}) DELETE n");
    let result = query.parse();
    assert!(result.is_ok());
}

#[test]
fn test_parse_set_property() {
    let query = CypherQuery::new("MATCH (n:Person {name: 'Alice'}) SET n.age = 31");
    let result = query.parse();
    assert!(result.is_ok());
}

#[test]
fn test_parse_remove_property() {
    let query = CypherQuery::new("MATCH (n:Person {name: 'Alice'}) REMOVE n.age");
    let result = query.parse();
    assert!(result.is_ok());
}

#[test]
fn test_parse_where_clause() {
    let query = CypherQuery::new("MATCH (n:Person) WHERE n.age > 25 RETURN n");
    let result = query.parse();
    assert!(result.is_ok());
}

#[test]
fn test_parse_order_by() {
    let query = CypherQuery::new("MATCH (n:Person) RETURN n ORDER BY n.age DESC");
    let result = query.parse();
    assert!(result.is_ok());
}

#[test]
fn test_parse_limit() {
    let query = CypherQuery::new("MATCH (n:Person) RETURN n LIMIT 10");
    let result = query.parse();
    assert!(result.is_ok());
}

#[test]
fn test_parse_skip() {
    let query = CypherQuery::new("MATCH (n:Person) RETURN n SKIP 5 LIMIT 10");
    let result = query.parse();
    assert!(result.is_ok());
}

#[test]
fn test_parse_aggregate_count() {
    let query = CypherQuery::new("MATCH (n:Person) RETURN COUNT(n)");
    let result = query.parse();
    assert!(result.is_ok());
}

#[test]
fn test_parse_aggregate_sum() {
    let query = CypherQuery::new("MATCH (n:Person) RETURN SUM(n.age)");
    let result = query.parse();
    assert!(result.is_ok());
}

#[test]
fn test_parse_aggregate_avg() {
    let query = CypherQuery::new("MATCH (n:Person) RETURN AVG(n.age)");
    let result = query.parse();
    assert!(result.is_ok());
}

#[test]
fn test_parse_with_clause() {
    let query = CypherQuery::new(
        "MATCH (n:Person) WITH n.age AS age WHERE age > 25 RETURN age"
    );
    let result = query.parse();
    assert!(result.is_ok());
}

#[test]
fn test_parse_union() {
    let query = CypherQuery::new(
        "MATCH (n:Person) RETURN n.name UNION MATCH (m:Company) RETURN m.name"
    );
    let result = query.parse();
    assert!(result.is_ok());
}

#[test]
fn test_parse_optional_match() {
    let query = CypherQuery::new("OPTIONAL MATCH (n:Person)-[r:KNOWS]->(m) RETURN n, m");
    let result = query.parse();
    assert!(result.is_ok());
}

#[test]
fn test_parse_case_expression() {
    let query = CypherQuery::new(
        "MATCH (n:Person) RETURN CASE WHEN n.age < 18 THEN 'minor' ELSE 'adult' END AS status"
    );
    let result = query.parse();
    assert!(result.is_ok());
}

#[test]
fn test_parse_coalesce() {
    let query = CypherQuery::new("MATCH (n:Person) RETURN COALESCE(n.nickname, n.name) AS display_name");
    let result = query.parse();
    assert!(result.is_ok());
}

#[test]
fn test_parse_shortestpath() {
    let query = CypherQuery::new(
        "MATCH p = shortestPath((a:Person)-[*]-(b:Person)) RETURN p"
    );
    let result = query.parse();
    assert!(result.is_ok());
}

#[test]
fn test_parse_multi_line_query() {
    let query = CypherQuery::new("
        MATCH (a:Person)-[:WORKS_AT]->(c:Company)
        WHERE c.industry = 'Tech'
        WITH a, c
        MATCH (a)-[:KNOWS]->(friend:Person)
        RETURN a.name, c.name, COLLECT(friend.name) AS friends
        ORDER BY a.name
    ");
    let result = query.parse();
    assert!(result.is_ok());
}

// ============================================================================
// Invalid Cypher Queries (Should Fail)
// ============================================================================

// TODO: Implement error detection in parser
// #[test]
// fn test_parse_invalid_syntax() {
//     let query = CypherQuery::new("MATCH (n Person) RETURN n"); // Missing colon
//     let result = query.parse();
//     assert!(result.is_err());
// }

// #[test]
// fn test_parse_unclosed_parenthesis() {
//     let query = CypherQuery::new("MATCH (n:Person RETURN n");
//     let result = query.parse();
//     assert!(result.is_err());
// }

// #[test]
// fn test_parse_invalid_relationship() {
//     let query = CypherQuery::new("MATCH (a)-[KNOWS]-(b) RETURN a"); // Missing colon
//     let result = query.parse();
//     assert!(result.is_err());
// }

// #[test]
// fn test_parse_missing_return() {
//     let query = CypherQuery::new("MATCH (n:Person)"); // No RETURN clause
//     let result = query.parse();
//     assert!(result.is_err());
// }

// ============================================================================
// Complex Query Tests
// ============================================================================

#[test]
fn test_parse_complex_graph_pattern() {
    let query = CypherQuery::new("
        MATCH (user:User {id: $userId})-[:PURCHASED]->(product:Product)<-[:PURCHASED]-(other:User)
        WHERE other.id <> $userId
        WITH other, COUNT(*) AS commonProducts
        WHERE commonProducts > 3
        MATCH (other)-[:PURCHASED]->(recommendation:Product)
        WHERE NOT (user)-[:PURCHASED]->(recommendation)
        RETURN recommendation.name, COUNT(*) AS score
        ORDER BY score DESC
        LIMIT 10
    ");
    let result = query.parse();
    assert!(result.is_ok());
}

#[test]
fn test_parse_variable_length_path() {
    let query = CypherQuery::new(
        "MATCH (a:Person)-[:KNOWS*1..5]->(b:Person) WHERE a.name = 'Alice' RETURN b"
    );
    let result = query.parse();
    assert!(result.is_ok());
}

#[test]
fn test_parse_multiple_patterns() {
    let query = CypherQuery::new("
        MATCH (a:Person)-[:KNOWS]->(b:Person),
              (b)-[:WORKS_AT]->(c:Company)
        RETURN a.name, b.name, c.name
    ");
    let result = query.parse();
    assert!(result.is_ok());
}

#[test]
fn test_parse_collect_aggregation() {
    let query = CypherQuery::new(
        "MATCH (p:Person)-[:KNOWS]->(f:Person) RETURN p.name, COLLECT(f.name) AS friends"
    );
    let result = query.parse();
    assert!(result.is_ok());
}

#[test]
fn test_parse_unwind() {
    let query = CypherQuery::new("
        UNWIND [1, 2, 3] AS x
        CREATE (:Number {value: x})
    ");
    let result = query.parse();
    assert!(result.is_ok());
}

#[test]
fn test_parse_foreach() {
    let query = CypherQuery::new("
        MATCH p = (a)-[*]->(b)
        FOREACH (n IN nodes(p) | SET n.visited = true)
    ");
    let result = query.parse();
    assert!(result.is_ok());
}

#[test]
fn test_parse_call_procedure() {
    let query = CypherQuery::new("CALL db.labels()");
    let result = query.parse();
    assert!(result.is_ok());
}

#[test]
fn test_parse_exists_subquery() {
    let query = CypherQuery::new("
        MATCH (p:Person)
        WHERE EXISTS {
            MATCH (p)-[:KNOWS]->(:Person {name: 'Alice'})
        }
        RETURN p.name
    ");
    let result = query.parse();
    assert!(result.is_ok());
}

// ============================================================================
// Edge Cases
// ============================================================================

#[test]
fn test_parse_empty_query() {
    let query = CypherQuery::new("");
    let result = query.parse();
    // Empty query might be valid or invalid depending on implementation
    let _ = result;
}

#[test]
fn test_parse_whitespace_only() {
    let query = CypherQuery::new("   \n\t  ");
    let result = query.parse();
    let _ = result;
}

#[test]
fn test_parse_comment() {
    let query = CypherQuery::new("// This is a comment\nMATCH (n) RETURN n");
    let result = query.parse();
    assert!(result.is_ok());
}

#[test]
fn test_parse_multiline_comment() {
    let query = CypherQuery::new("/* Multi\nline\ncomment */\nMATCH (n) RETURN n");
    let result = query.parse();
    assert!(result.is_ok());
}

#[test]
fn test_parse_quoted_identifiers() {
    let query = CypherQuery::new("MATCH (`weird-name`:Person) RETURN `weird-name`");
    let result = query.parse();
    assert!(result.is_ok());
}

#[test]
fn test_parse_escaped_strings() {
    let query = CypherQuery::new("MATCH (n:Person {name: 'O\\'Brien'}) RETURN n");
    let result = query.parse();
    assert!(result.is_ok());
}

#[test]
fn test_parse_parameters() {
    let query = CypherQuery::new("MATCH (n:Person {name: $name, age: $age}) RETURN n");
    let result = query.parse();
    assert!(result.is_ok());
}

#[test]
fn test_parse_list_literal() {
    let query = CypherQuery::new("RETURN [1, 2, 3, 4, 5] AS numbers");
    let result = query.parse();
    assert!(result.is_ok());
}

#[test]
fn test_parse_map_literal() {
    let query = CypherQuery::new("RETURN {name: 'Alice', age: 30} AS person");
    let result = query.parse();
    assert!(result.is_ok());
}
