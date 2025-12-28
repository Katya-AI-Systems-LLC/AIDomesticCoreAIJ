"""
Locust Load Testing Suite for AI Platform API
Tests various endpoints under load to identify performance bottlenecks
"""

import os
import json
import time
import random
from locust import HttpUser, task, between, events
from locust.clients import RequestException
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
API_BASE_URL = os.getenv('API_BASE_URL', 'http://localhost:8000')
API_TOKEN = os.getenv('API_TOKEN', 'test_token')
TEST_MODE = os.getenv('TEST_MODE', 'basic')  # basic, intermediate, advanced


class BaseUser(HttpUser):
    """Base user class with common setup"""
    
    wait_time = between(1, 3)
    
    def on_start(self):
        """Initialize test user"""
        self.headers = {
            'Authorization': f'Bearer {API_TOKEN}',
            'Content-Type': 'application/json'
        }
        self.user_id = f"test_user_{random.randint(1000, 9999)}"
        self.project_id = None
        self.optimization_job_id = None
        self.vision_job_id = None
        
    def on_stop(self):
        """Cleanup after test"""
        logger.info(f"Test user {self.user_id} finished")


class HealthCheckUser(BaseUser):
    """Users performing health check operations"""
    
    wait_time = between(0.5, 2)
    
    @task(100)
    def health_check(self):
        """Health check endpoint - high frequency"""
        with self.client.get(
            '/api/v1/health',
            headers=self.headers,
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.mark_success()
            else:
                response.mark_failure(f"Status: {response.status_code}")
    
    @task(50)
    def status_check(self):
        """Status endpoint - medium frequency"""
        with self.client.get(
            '/api/v1/status',
            headers=self.headers,
            catch_response=True
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if 'status' in data:
                    response.mark_success()
                else:
                    response.mark_failure("Missing status field")
            else:
                response.mark_failure(f"Status: {response.status_code}")


class OptimizationUser(BaseUser):
    """Users submitting and monitoring optimization jobs"""
    
    wait_time = between(2, 5)
    
    @task(40)
    def submit_optimization(self):
        """Submit optimization job"""
        payload = {
            "problem_type": random.choice(["TSP", "MAXCUT", "QAOA"]),
            "problem_data": {
                "cities": random.randint(5, 20),
                "graph_size": random.randint(10, 50)
            },
            "parameters": {
                "optimizer": random.choice(["VQE", "QAOA", "Annealing"]),
                "iterations": random.randint(100, 1000)
            }
        }
        
        with self.client.post(
            '/api/v1/optimize',
            json=payload,
            headers=self.headers,
            catch_response=True
        ) as response:
            if response.status_code == 202:
                try:
                    data = response.json()
                    self.optimization_job_id = data.get('job_id')
                    response.mark_success()
                except:
                    response.mark_failure("Invalid JSON response")
            else:
                response.mark_failure(f"Status: {response.status_code}")
    
    @task(30)
    def get_optimization_status(self):
        """Get optimization job status"""
        if not self.optimization_job_id:
            return
        
        with self.client.get(
            f'/api/v1/optimize/{self.optimization_job_id}',
            headers=self.headers,
            catch_response=True
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if 'status' in data:
                    response.mark_success()
                else:
                    response.mark_failure("Missing status field")
            else:
                response.mark_failure(f"Status: {response.status_code}")
    
    @task(5)
    def cancel_optimization(self):
        """Cancel optimization job"""
        if not self.optimization_job_id:
            return
        
        with self.client.post(
            f'/api/v1/optimize/{self.optimization_job_id}/cancel',
            headers=self.headers,
            catch_response=True
        ) as response:
            if response.status_code in [200, 204]:
                response.mark_success()
                self.optimization_job_id = None
            else:
                response.mark_failure(f"Status: {response.status_code}")


class VisionUser(BaseUser):
    """Users submitting vision analysis jobs"""
    
    wait_time = between(3, 7)
    
    @task(30)
    def submit_vision_analysis(self):
        """Submit vision analysis job"""
        # Use base64 encoded dummy image for testing
        dummy_image = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
        
        payload = {
            "image": dummy_image,
            "analysis_type": random.choice(["DETECTION", "CLASSIFICATION", "SEGMENTATION"]),
            "models": random.sample(["yolo", "resnet", "vit"], k=random.randint(1, 3))
        }
        
        with self.client.post(
            '/api/v1/vision/analyze',
            json=payload,
            headers=self.headers,
            catch_response=True
        ) as response:
            if response.status_code == 202:
                try:
                    data = response.json()
                    self.vision_job_id = data.get('job_id')
                    response.mark_success()
                except:
                    response.mark_failure("Invalid JSON response")
            else:
                response.mark_failure(f"Status: {response.status_code}")
    
    @task(20)
    def get_vision_status(self):
        """Get vision analysis status"""
        if not self.vision_job_id:
            return
        
        with self.client.get(
            f'/api/v1/vision/analyze/{self.vision_job_id}',
            headers=self.headers,
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.mark_success()
            else:
                response.mark_failure(f"Status: {response.status_code}")


class ProjectManagementUser(BaseUser):
    """Users managing projects"""
    
    wait_time = between(2, 4)
    
    @task(25)
    def list_projects(self):
        """List user projects"""
        with self.client.get(
            '/api/v1/projects?limit=10&offset=0',
            headers=self.headers,
            catch_response=True
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if 'projects' in data:
                    response.mark_success()
                else:
                    response.mark_failure("Missing projects field")
            else:
                response.mark_failure(f"Status: {response.status_code}")
    
    @task(20)
    def create_project(self):
        """Create new project"""
        payload = {
            "name": f"Project_{random.randint(1000, 9999)}",
            "description": "Load test project",
            "visibility": random.choice(["PRIVATE", "PUBLIC"])
        }
        
        with self.client.post(
            '/api/v1/projects',
            json=payload,
            headers=self.headers,
            catch_response=True
        ) as response:
            if response.status_code == 201:
                try:
                    data = response.json()
                    self.project_id = data.get('project_id')
                    response.mark_success()
                except:
                    response.mark_failure("Invalid JSON response")
            else:
                response.mark_failure(f"Status: {response.status_code}")
    
    @task(15)
    def update_project(self):
        """Update project"""
        if not self.project_id:
            return
        
        payload = {
            "name": f"Updated_Project_{random.randint(1000, 9999)}",
            "description": "Updated description"
        }
        
        with self.client.put(
            f'/api/v1/projects/{self.project_id}',
            json=payload,
            headers=self.headers,
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.mark_success()
            else:
                response.mark_failure(f"Status: {response.status_code}")
    
    @task(5)
    def delete_project(self):
        """Delete project"""
        if not self.project_id:
            return
        
        with self.client.delete(
            f'/api/v1/projects/{self.project_id}',
            headers=self.headers,
            catch_response=True
        ) as response:
            if response.status_code in [200, 204]:
                response.mark_success()
                self.project_id = None
            else:
                response.mark_failure(f"Status: {response.status_code}")


class InferenceUser(BaseUser):
    """Users running inference"""
    
    wait_time = between(2, 6)
    
    @task(35)
    def run_inference(self):
        """Run inference on model"""
        payload = {
            "model": random.choice(["bert-base", "gpt2", "vit-base"]),
            "input": {
                "text": "This is a test input for inference"
            }
        }
        
        with self.client.post(
            '/api/v1/infer/predict',
            json=payload,
            headers=self.headers,
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.mark_success()
            else:
                response.mark_failure(f"Status: {response.status_code}")
    
    @task(25)
    def list_models(self):
        """List available models"""
        with self.client.get(
            '/api/v1/models',
            headers=self.headers,
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.mark_success()
            else:
                response.mark_failure(f"Status: {response.status_code}")


class GraphQLUser(BaseUser):
    """Users making GraphQL queries"""
    
    wait_time = between(1, 4)
    
    @task(50)
    def graphql_health_query(self):
        """GraphQL health query"""
        payload = {
            "query": "{ health { status uptime } }"
        }
        
        with self.client.post(
            '/graphql',
            json=payload,
            headers=self.headers,
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.mark_success()
            else:
                response.mark_failure(f"Status: {response.status_code}")
    
    @task(30)
    def graphql_projects_query(self):
        """GraphQL projects query"""
        payload = {
            "query": """
                query {
                    projects(first: 10) {
                        edges {
                            node {
                                id
                                name
                                createdAt
                            }
                        }
                    }
                }
            """
        }
        
        with self.client.post(
            '/graphql',
            json=payload,
            headers=self.headers,
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.mark_success()
            else:
                response.mark_failure(f"Status: {response.status_code}")


class StressTestUser(BaseUser):
    """User for stress testing - rapid requests"""
    
    wait_time = between(0.1, 0.5)
    
    @task(100)
    def rapid_health_checks(self):
        """Rapid health checks for stress testing"""
        with self.client.get(
            '/api/v1/health',
            headers=self.headers,
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.mark_success()
            else:
                response.mark_failure(f"Status: {response.status_code}")


# Event handlers for monitoring
@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    logger.info(f"Load test started: {environment.host}")


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    logger.info("Load test finished")
    
    # Print statistics
    if hasattr(environment.stats, 'total'):
        logger.info(f"Total requests: {environment.stats.total.num_requests}")
        logger.info(f"Total failures: {environment.stats.total.num_failures}")
        logger.info(f"Average response time: {environment.stats.total.avg_response_time}ms")


@events.quitting.add_listener
def on_quitting(environment, **kwargs):
    logger.info("Test quitting")


# Default setup
if __name__ == "__main__":
    # Run with: locust -f locustfile.py --host=http://localhost:8000
    pass
