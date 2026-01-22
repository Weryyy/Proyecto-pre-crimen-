// Neo4j Database Schema for Pre-Crime Prediction System
// This schema defines the graph structure for citizens, locations, and their relationships

// Create constraints for unique IDs
CREATE CONSTRAINT citizen_id IF NOT EXISTS FOR (c:Citizen) REQUIRE c.id IS UNIQUE;
CREATE CONSTRAINT location_id IF NOT EXISTS FOR (l:Location) REQUIRE l.id IS UNIQUE;

// Create indexes for better query performance
CREATE INDEX citizen_risk IF NOT EXISTS FOR (c:Citizen) ON (c.risk_seed);
CREATE INDEX location_crime_rate IF NOT EXISTS FOR (l:Location) ON (l.crime_rate);

// Node: Citizen
// Properties:
//   - id: unique identifier
//   - name: citizen name
//   - age: age of the citizen
//   - risk_seed: risk score using Beta distribution (evolves over time)
//   - occupation: citizen's occupation
//   - created_at: timestamp of creation

// Node: Location
// Properties:
//   - id: unique identifier
//   - name: location name
//   - latitude: geographical latitude
//   - longitude: geographical longitude
//   - crime_rate: historical crime rate
//   - area_type: type of area (residential, commercial, industrial)

// Relationship: INTERACTS_WITH
// Properties:
//   - timestamp: when the interaction occurred
//   - interaction_type: type of interaction (friend, family, colleague, conflict)
//   - frequency: how often they interact
//   - strength: strength of the relationship (0-1)

// Relationship: MOVES_TO
// Properties:
//   - timestamp: when the movement occurred
//   - duration: how long they stayed (in minutes)
//   - purpose: purpose of visit

// Example query to create sample data:
// CREATE (c1:Citizen {id: 'C001', name: 'John Doe', age: 35, risk_seed: 0.15, occupation: 'Engineer', created_at: datetime()})
// CREATE (c2:Citizen {id: 'C002', name: 'Jane Smith', age: 28, risk_seed: 0.05, occupation: 'Teacher', created_at: datetime()})
// CREATE (l1:Location {id: 'L001', name: 'Downtown Plaza', latitude: 40.7589, longitude: -73.9851, crime_rate: 0.12, area_type: 'commercial'})
// CREATE (c1)-[:INTERACTS_WITH {timestamp: datetime(), interaction_type: 'friend', frequency: 10, strength: 0.8}]->(c2)
// CREATE (c1)-[:MOVES_TO {timestamp: datetime(), duration: 120, purpose: 'shopping'}]->(l1)
