# GraphQL Guide - AI Platform

## Overview

GraphQL provides a flexible, strongly-typed query language for the AI Platform API. Use GraphQL when you need precise control over response data, real-time subscriptions, or batch operations.

## Why GraphQL?

- **Precise Data Fetching**: Request only the fields you need
- **Single Endpoint**: All operations through one URL
- **Strong Typing**: Full schema introspection and validation
- **Batch Queries**: Multiple operations in one request
- **Real-Time**: Subscriptions for live updates
- **Efficient**: Reduce network overhead with nested queries

## Getting Started

### Endpoint

```
https://api.aiplatform.io/graphql
```

### Authentication

Include your API key in the Authorization header:

```bash
curl -H "Authorization: Bearer YOUR_API_KEY" \
     -H "Content-Type: application/json" \
     https://api.aiplatform.io/graphql
```

### First Query

```graphql
query {
  health {
    status
    version
  }
}
```

## Query Examples

### Health and Status

#### Check Service Health

```graphql
query {
  health {
    status
    timestamp
    version
    uptimeSeconds
  }
}
```

Response:
```json
{
  "data": {
    "health": {
      "status": "healthy",
      "timestamp": "2024-01-15T10:30:00Z",
      "version": "1.0.0",
      "uptimeSeconds": 3600
    }
  }
}
```

#### Detailed Service Status

```graphql
query {
  status {
    status
    timestamp
    services {
      api
      quantum
      vision
      database
      cache
    }
  }
}
```

### Optimization Jobs

#### Get Single Job

```graphql
query GetOptimization($jobId: ID!) {
  optimizationJob(id: $jobId) {
    id
    status
    problemType
    numQubits
    createdAt
    estimatedCompletion
    progress
  }
}
```

Variables:
```json
{
  "jobId": "550e8400-e29b-41d4-a716-446655440000"
}
```

#### List Jobs with Pagination

```graphql
query ListOptimizations($first: Int, $after: String) {
  optimizationJobs(pagination: {first: $first, after: $after}) {
    edges {
      node {
        id
        status
        problemType
        createdAt
      }
      cursor
    }
    pageInfo {
      hasNextPage
      hasPreviousPage
      endCursor
    }
    totalCount
  }
}
```

Variables:
```json
{
  "first": 10,
  "after": null
}
```

### Vision Analysis

#### Get Analysis Results

```graphql
query GetAnalysis($id: ID!) {
  visionAnalysis(id: $id) {
    id
    detections {
      className
      confidence
      boundingBox {
        x
        y
        width
        height
      }
    }
    classifications {
      label
      confidence
    }
    ocrText
    sceneDescription
    processedAt
  }
}
```

#### List Vision Jobs

```graphql
query ListVisionJobs($status: JobStatus) {
  visionAnalysisJobs(status: $status) {
    id
    detections {
      className
      confidence
    }
    classifications {
      label
    }
    processedAt
  }
}
```

### Models

#### Get Model Details

```graphql
query GetModel($modelId: String!) {
  model(id: $modelId) {
    id
    name
    description
    taskType
    framework
    version
    accuracy
    inputSchema
    outputSchema
    createdAt
    updatedAt
  }
}
```

#### List All Models

```graphql
query ListModels($taskType: TaskType, $framework: String) {
  models(taskType: $taskType, framework: $framework) {
    id
    name
    taskType
    framework
    version
    accuracy
  }
}
```

### Projects

#### Get Project

```graphql
query GetProject($projectId: ID!) {
  project(id: $projectId) {
    id
    name
    description
    owner {
      id
      name
      email
    }
    members {
      id
      name
      email
      role
    }
    createdAt
    updatedAt
    tags
  }
}
```

#### List User Projects

```graphql
query ListProjects {
  projects {
    id
    name
    description
    createdAt
    tags
  }
}
```

### Current User

```graphql
query {
  currentUser {
    id
    email
    name
    role
    tier
    createdAt
  }
}
```

### Search

```graphql
query Search($query: String!, $types: [String!]) {
  search(query: $query, types: $types) {
    ... on Project {
      id
      name
      createdAt
    }
    ... on MLModel {
      id
      name
      taskType
    }
    ... on OptimizationJob {
      id
      status
      problemType
    }
    ... on VisionAnalysis {
      id
      sceneDescription
      processedAt
    }
  }
}
```

## Mutation Examples

### Quantum Optimization

#### Submit Optimization

```graphql
mutation SubmitOptimization($input: OptimizationInput!) {
  submitOptimization(input: $input) {
    id
    status
    createdAt
    estimatedCompletion
  }
}
```

Variables:
```json
{
  "input": {
    "problemType": "MAXCUT",
    "numQubits": 4,
    "graph": {
      "nodes": [0, 1, 2, 3],
      "edges": [[0, 1], [1, 2], [2, 3], [3, 0]]
    }
  }
}
```

#### Cancel Optimization

```graphql
mutation CancelOptimization($jobId: ID!) {
  cancelOptimization(jobId: $jobId)
}
```

### Vision Analysis

#### Submit Image Analysis

```graphql
mutation AnalyzeImage($input: VisionAnalysisInput!) {
  submitVisionAnalysis(input: $input) {
    id
    detections {
      className
      confidence
    }
    classifications {
      label
      confidence
    }
    processedAt
  }
}
```

#### Submit Batch Analysis

```graphql
mutation BatchAnalysis($input: BatchAnalysisInput!) {
  submitBatchAnalysis(input: $input) {
    id
    status
    totalImages
    processedImages
  }
}
```

### Federated Learning

#### Start Training

```graphql
mutation StartTraining($input: FederatedTrainingInput!) {
  startFederatedTraining(input: $input) {
    id
    status
    modelName
    numClients
    createdAt
  }
}
```

Variables:
```json
{
  "input": {
    "modelName": "federated-classifier",
    "numClients": 10,
    "epochs": 5,
    "privacyBudget": 1.0
  }
}
```

#### Update Training

```graphql
mutation UpdateTraining($jobId: ID!, $parameters: JSON!) {
  updateFederatedTraining(jobId: $jobId, parameters: $parameters) {
    id
    status
  }
}
```

### Inference

#### Run Prediction

```graphql
mutation Predict($input: InferenceInput!) {
  runInference(input: $input) {
    prediction
    confidence
    latencyMs
    modelVersion
  }
}
```

Variables:
```json
{
  "input": {
    "modelId": "bert-base-uncased",
    "input": {
      "text": "Hello, world!"
    }
  }
}
```

#### Batch Inference

```graphql
mutation BatchInference($modelId: String!, $inputs: [JSON!]!) {
  submitBatchInference(modelId: $modelId, inputs: $inputs) {
    id
    status
    submittedAt
  }
}
```

### Project Management

#### Create Project

```graphql
mutation CreateProject($input: ProjectInput!) {
  createProject(input: $input) {
    id
    name
    description
    createdAt
  }
}
```

Variables:
```json
{
  "input": {
    "name": "My ML Project",
    "description": "Description of project"
  }
}
```

#### Update Project

```graphql
mutation UpdateProject($projectId: ID!, $input: ProjectUpdateInput!) {
  updateProject(id: $projectId, input: $input) {
    id
    name
    description
    updatedAt
  }
}
```

#### Delete Project

```graphql
mutation DeleteProject($projectId: ID!) {
  deleteProject(id: $projectId)
}
```

### Model Management

#### Fine-tune Model

```graphql
mutation FinetuneModel(
  $modelId: String!
  $datasetUrl: String!
  $epochs: Int!
) {
  finetuneModel(
    modelId: $modelId
    datasetUrl: $datasetUrl
    epochs: $epochs
  ) {
    id
    modelId
    status
    datasetSize
    epochs
  }
}
```

#### Deploy Model

```graphql
mutation DeployModel($modelId: String!, $replicas: Int!) {
  deployModel(modelId: $modelId, replicas: $replicas) {
    id
    modelId
    status
    replicas
    endpoint
  }
}
```

### API Key Management

#### Create API Key

```graphql
mutation CreateKey($input: ApiKeyInput!) {
  createApiKey(input: $input) {
    id
    name
    prefix
    createdAt
    expiresAt
  }
}
```

Variables:
```json
{
  "input": {
    "name": "Production Key",
    "expiresIn": 31536000
  }
}
```

#### Revoke API Key

```graphql
mutation RevokeKey($keyId: ID!) {
  revokeApiKey(keyId: $keyId)
}
```

## Subscription Examples

### Real-Time Optimization Progress

```graphql
subscription OnOptimizationProgress($jobId: ID!) {
  optimizationProgress(jobId: $jobId) {
    jobId
    solution
    objectiveValue
    approximationRatio
    executionTime
  }
}
```

### Vision Analysis Progress

```graphql
subscription OnVisionProgress($jobId: ID!) {
  visionAnalysisProgress(jobId: $jobId) {
    id
    detections {
      className
      confidence
    }
    processedAt
  }
}
```

### Training Progress

```graphql
subscription OnTrainingProgress($jobId: ID!) {
  trainingProgress(jobId: $jobId) {
    jobId
    round
    accuracy
    loss
    timestamp
  }
}
```

### Inference Completion

```graphql
subscription OnInferenceComplete($jobId: ID!) {
  inferenceCompleted(jobId: $jobId) {
    prediction
    confidence
    latencyMs
  }
}
```

### Model Deployment Progress

```graphql
subscription OnDeploymentProgress($deploymentId: ID!) {
  deploymentProgress(deploymentId: $deploymentId) {
    id
    modelId
    status
    replicas
  }
}
```

## Advanced Patterns

### Batch Queries

Fetch multiple resources efficiently:

```graphql
query BatchQuery {
  # Multiple queries in one request
  health {
    status
  }
  
  myProjects: projects {
    id
    name
  }
  
  currentUser: currentUser {
    id
    email
    tier
  }
}
```

### Aliases

Fetch the same query with different parameters:

```graphql
query CompareModels {
  bestModel: model(id: "model-1") {
    name
    accuracy
  }
  
  alternativeModel: model(id: "model-2") {
    name
    accuracy
  }
}
```

### Fragments

Reuse query fragments:

```graphql
fragment ProjectDetails on Project {
  id
  name
  description
  createdAt
  members {
    id
    name
  }
}

query GetProjects {
  projects {
    ...ProjectDetails
  }
}
```

### Error Handling

GraphQL returns status 200 even with errors. Check the `errors` field:

```python
import requests
import json

def graphql_query(query, variables=None):
    response = requests.post(
        'https://api.aiplatform.io/graphql',
        headers={
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        },
        json={
            'query': query,
            'variables': variables or {}
        }
    )
    
    result = response.json()
    
    # Check for GraphQL errors
    if 'errors' in result:
        for error in result['errors']:
            print(f"GraphQL Error: {error['message']}")
            if 'locations' in error:
                print(f"  at line {error['locations'][0]['line']}")
    
    return result.get('data')
```

## Type Introspection

Discover the schema:

```graphql
query Schema {
  __schema {
    types {
      name
      kind
      description
      fields {
        name
        type {
          name
          kind
        }
      }
    }
  }
}
```

## Rate Limiting

GraphQL queries have similar rate limits to REST:

- Standard: 1,000 requests/hour
- Premium: 10,000 requests/hour
- Enterprise: Unlimited

Complex queries count as multiple tokens based on their computational cost.

## Best Practices

1. **Use Fragments** for reusable query parts
2. **Request Only What You Need** - be specific with fields
3. **Use Variables** for dynamic values
4. **Handle Errors** - check the errors field
5. **Batch Operations** - combine multiple operations
6. **Cache Results** - use client-side caching for frequently accessed data
7. **Monitor Query Complexity** - complex queries may count as more tokens

## Tools

### GraphQL IDE

Try queries at:
- Apollo Studio: https://studio.apollographql.com
- Insomnia: https://insomnia.rest
- Postman: https://postman.com
- GraphQL Playground: https://www.apollographql.com/docs/apollo-server/testing/graphql-playground/

### Python Client

```bash
pip install gql[all]
```

```python
from gql import gql, Client
from gql.transport.requests import RequestsHTTPTransport

transport = RequestsHTTPTransport(
    url='https://api.aiplatform.io/graphql',
    headers={'Authorization': f'Bearer {api_key}'}
)

client = Client(transport=transport)

query = gql('''
    query {
        health {
            status
        }
    }
''')

result = client.execute(query)
```

### JavaScript Client

```bash
npm install @apollo/client graphql
```

```javascript
import ApolloClient from '@apollo/client';
import { gql } from '@apollo/client';

const client = new ApolloClient({
  uri: 'https://api.aiplatform.io/graphql',
  headers: {
    authorization: `Bearer ${apiKey}`
  }
});

const query = gql`
  query {
    health {
      status
    }
  }
`;

client.query({ query }).then(result => console.log(result));
```

## Conclusion

GraphQL provides a powerful, flexible way to interact with the AI Platform API. For REST API documentation, see the [API Guide](API_GUIDE.md).
