// Jenkins Declarative Pipeline for AI Platform
// Builds, tests, scans, and deploys the application

pipeline {
    agent any

    options {
        buildDiscarder(logRotator(numToKeepStr: '30'))
        timeout(time: 1, unit: 'HOURS')
        timestamps()
        ansiColor('xterm')
    }

    parameters {
        choice(
            name: 'ENVIRONMENT',
            choices: ['dev', 'staging', 'production'],
            description: 'Deployment environment'
        )
        booleanParam(
            name: 'SKIP_TESTS',
            defaultValue: false,
            description: 'Skip running tests'
        )
        string(
            name: 'DOCKER_REGISTRY',
            defaultValue: 'docker.io',
            description: 'Docker registry URL'
        )
    }

    environment {
        DOCKER_IMAGE = "${params.DOCKER_REGISTRY}/aiplatform/sdk"
        DOCKER_TAG = "${BUILD_NUMBER}"
        FULL_IMAGE = "${DOCKER_IMAGE}:${DOCKER_TAG}"
        REGISTRY_CREDENTIALS = credentials('docker-registry-credentials')
        GIT_COMMIT_SHORT = sh(script: "git rev-parse --short HEAD", returnStdout: true).trim()
        BUILD_TIMESTAMP = sh(script: "date -u +'%Y-%m-%dT%H:%M:%SZ'", returnStdout: true).trim()
    }

    stages {
        stage('Checkout') {
            steps {
                script {
                    echo "🔄 Checking out code from ${GIT_BRANCH}..."
                }
                checkout scm
                script {
                    echo "✅ Code checked out successfully"
                    echo "Git Commit: ${GIT_COMMIT_SHORT}"
                    echo "Build: ${BUILD_NUMBER}"
                }
            }
        }

        stage('Install Dependencies') {
            steps {
                script {
                    echo "📦 Installing Python dependencies..."
                    sh '''
                        python -m pip install --upgrade pip setuptools wheel
                        pip install -r requirements.txt
                        pip install pytest pytest-cov pytest-xdist flake8 bandit safety
                    '''
                }
            }
        }

        stage('Code Quality') {
            parallel {
                stage('Lint') {
                    steps {
                        script {
                            echo "🔍 Running flake8 linter..."
                            sh '''
                                flake8 aiplatform/ --max-line-length=100 --count --statistics || true
                            '''
                        }
                    }
                }

                stage('Security Scan') {
                    steps {
                        script {
                            echo "🔐 Running security checks..."
                            sh '''
                                # Bandit for Python security issues
                                bandit -r aiplatform/ -f json -o bandit-report.json || true
                                
                                # Safety check for vulnerable dependencies
                                safety check --json > safety-report.json || true
                            '''
                        }
                    }
                }

                stage('Dependency Check') {
                    steps {
                        script {
                            echo "📋 Checking dependencies..."
                            sh '''
                                pip-audit --desc > pip-audit-report.txt || true
                            '''
                        }
                    }
                }
            }
        }

        stage('Unit Tests') {
            when {
                expression { !params.SKIP_TESTS }
            }
            steps {
                script {
                    echo "🧪 Running unit tests..."
                    sh '''
                        pytest tests/ \
                            -v \
                            --cov=aiplatform \
                            --cov-report=xml \
                            --cov-report=html \
                            --junitxml=junit.xml \
                            -n auto
                    '''
                }
            }
        }

        stage('Integration Tests') {
            when {
                expression { !params.SKIP_TESTS }
            }
            steps {
                script {
                    echo "🔗 Running integration tests..."
                    sh '''
                        pytest tests/integration_tests.py \
                            -v \
                            --timeout=300 \
                            --junitxml=integration-junit.xml
                    '''
                }
            }
        }

        stage('Build Docker Image') {
            steps {
                script {
                    echo "🐳 Building Docker image: ${FULL_IMAGE}..."
                    sh '''
                        docker build \
                            -t ${FULL_IMAGE} \
                            -t ${DOCKER_IMAGE}:${ENVIRONMENT} \
                            -t ${DOCKER_IMAGE}:latest \
                            --build-arg BUILD_DATE=${BUILD_TIMESTAMP} \
                            --build-arg VCS_REF=${GIT_COMMIT_SHORT} \
                            --build-arg VERSION=${BUILD_NUMBER} \
                            -f Dockerfile .
                    '''
                    echo "✅ Docker image built successfully"
                }
            }
        }

        stage('Container Security Scan') {
            steps {
                script {
                    echo "🔐 Scanning Docker image for vulnerabilities..."
                    sh '''
                        # Install trivy if not present
                        which trivy || (curl -sfL https://raw.githubusercontent.com/aquasecurity/trivy/main/contrib/install.sh | sh -s -- -b /usr/local/bin)
                        
                        # Scan image
                        trivy image --exit-code 0 --severity HIGH,CRITICAL \
                            --format json -o trivy-report.json ${FULL_IMAGE} || true
                        
                        # Display summary
                        trivy image --exit-code 0 --severity HIGH,CRITICAL ${FULL_IMAGE}
                    '''
                }
            }
        }

        stage('Push Docker Image') {
            when {
                branch 'main'
            }
            steps {
                script {
                    echo "📤 Pushing Docker image to registry..."
                    sh '''
                        echo "${REGISTRY_CREDENTIALS_PSW}" | docker login -u "${REGISTRY_CREDENTIALS_USR}" --password-stdin ${DOCKER_REGISTRY}
                        
                        docker push ${FULL_IMAGE}
                        docker push ${DOCKER_IMAGE}:${ENVIRONMENT}
                        docker push ${DOCKER_IMAGE}:latest
                        
                        echo "✅ Docker image pushed successfully"
                    '''
                }
            }
        }

        stage('Deploy to Dev') {
            when {
                branch 'develop'
            }
            steps {
                script {
                    echo "🚀 Deploying to DEV environment..."
                    sh '''
                        kubectl config use-context dev-cluster
                        kubectl set image deployment/aiplatform-api \
                            aiplatform-api=${FULL_IMAGE} \
                            -n aiplatform \
                            --record
                        
                        kubectl rollout status deployment/aiplatform-api -n aiplatform
                    '''
                }
            }
        }

        stage('Deploy to Staging') {
            when {
                branch 'staging'
            }
            steps {
                script {
                    echo "🚀 Deploying to STAGING environment..."
                    sh '''
                        kubectl config use-context staging-cluster
                        
                        kubectl set image deployment/aiplatform-api \
                            aiplatform-api=${FULL_IMAGE} \
                            -n aiplatform \
                            --record
                        
                        kubectl rollout status deployment/aiplatform-api -n aiplatform
                        
                        # Run smoke tests
                        sleep 30
                        kubectl run smoke-test \
                            --image=${FULL_IMAGE} \
                            --restart=Never \
                            -n aiplatform \
                            -- pytest tests/smoke_tests.py
                    '''
                }
            }
        }

        stage('Deploy to Production') {
            when {
                branch 'main'
            }
            input {
                message "Deploy to PRODUCTION?"
                ok "Deploy"
                submitter "admin,devops"
            }
            steps {
                script {
                    echo "🚀 Deploying to PRODUCTION environment..."
                    sh '''
                        kubectl config use-context production-cluster
                        
                        # Use blue-green deployment
                        kubectl set image deployment/aiplatform-api-blue \
                            aiplatform-api=${FULL_IMAGE} \
                            -n aiplatform \
                            --record
                        
                        kubectl rollout status deployment/aiplatform-api-blue -n aiplatform
                        
                        # Run health checks
                        sleep 30
                        kubectl run health-check \
                            --image=${FULL_IMAGE} \
                            --restart=Never \
                            -n aiplatform \
                            -- pytest tests/health_check.py
                        
                        # Switch traffic to blue deployment
                        kubectl patch service aiplatform-api \
                            -n aiplatform \
                            -p '{"spec":{"selector":{"version":"blue"}}}'
                        
                        echo "✅ Production deployment completed"
                    '''
                }
            }
        }

        stage('Performance Tests') {
            when {
                branch 'main'
            }
            steps {
                script {
                    echo "⚡ Running performance tests..."
                    sh '''
                        pip install locust
                        locust -f tests/locustfile.py \
                            --host=https://api.aiplatform.com \
                            --users=100 \
                            --spawn-rate=10 \
                            --run-time=5m \
                            --headless
                    '''
                }
            }
        }
    }

    post {
        always {
            script {
                echo "📊 Collecting reports..."
                
                // Publish test results
                junit allowEmptyResults: true, testResults: '**/junit.xml'
                
                // Publish coverage
                publishHTML([
                    reportDir: 'htmlcov',
                    reportFiles: 'index.html',
                    reportName: 'Code Coverage'
                ])
                
                // Publish security reports
                publishHTML([
                    reportDir: '.',
                    reportFiles: 'pip-audit-report.txt',
                    reportName: 'Dependency Audit'
                ])
                
                // Archive logs
                archiveArtifacts artifacts: '*.json,*.xml,*.txt,htmlcov/**', allowEmptyArchive: true
                
                // Cleanup
                cleanWs deleteDirs: true, patterns: [[pattern: 'venv/**', type: 'INCLUDE']]
            }
        }

        success {
            script {
                echo "✅ Pipeline completed successfully"
                // Send success notification
                sh '''
                    # Slack notification example
                    # curl -X POST -H 'Content-type: application/json' \
                    #     --data "{\\"text\\":\\"✅ Build ${BUILD_NUMBER} successful\\"}" \
                    #     $SLACK_WEBHOOK_URL
                '''
            }
        }

        failure {
            script {
                echo "❌ Pipeline failed"
                // Send failure notification
                sh '''
                    # Slack notification example
                    # curl -X POST -H 'Content-type: application/json' \
                    #     --data "{\\"text\\":\\"❌ Build ${BUILD_NUMBER} failed\\"}" \
                    #     $SLACK_WEBHOOK_URL
                '''
            }
        }

        unstable {
            script {
                echo "⚠️  Pipeline unstable"
            }
        }
    }
}
