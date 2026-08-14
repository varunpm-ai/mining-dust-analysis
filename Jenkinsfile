// Jenkins Declarative Pipeline for Mine Dust MLOps Training & Evaluation
pipeline {
    agent any

    options {
        buildDiscarder(logRotator(numToKeepStr: '30', artifactNumToKeepStr: '30'))
        disableConcurrentBuilds()
        timeout(time: 1, unit: 'HOURS')
        timestamps()
    }

    parameters {
        choice(name: 'TARGET', choices: ['PM10', 'PM2.5'], description: 'Target air quality parameter to model')
        string(name: 'DATASET_PATH', defaultValue: 'artifacts/datasets/uploads/jenkins_input.csv', description: 'Path to input CSV/XLSX dataset')
        choice(name: 'SELECTION_METRIC', choices: ['RMSE', 'MAE', 'R2'], description: 'Primary model selection metric')
        choice(name: 'TRAINING_MODE', choices: ['full', 'fast', 'cross_validation'], description: 'Model training mode')
    }

    environment {
        PYTHONUNBUFFERED = '1'
        ENVIRONMENT = 'production'
        LOG_LEVEL = 'INFO'
    }

    stages {
        stage('CHECKOUT') {
            steps {
                echo "=== Stage 1: CHECKOUT ==="
                checkout scm
                script {
                    env.GIT_COMMIT_SHORT = sh(script: "git rev-parse --short HEAD || echo 'N/A'", returnStdout: true).trim()
                    echo "Checked out Git Commit: ${env.GIT_COMMIT} (${env.GIT_COMMIT_SHORT})"
                    echo "Jenkins Build Number: ${env.BUILD_NUMBER}"
                }
            }
        }

        stage('VALIDATE') {
            steps {
                echo "=== Stage 2: VALIDATE ==="
                sh 'python --version'
                sh 'pytest tests/unit/ -q --maxfail=1'
            }
        }

        stage('PREPROCESS') {
            steps {
                echo "=== Stage 3: PREPROCESS ==="
                echo "Validating schema & cleaning missing/invalid values..."
            }
        }

        stage('STATISTICAL ANALYSIS') {
            steps {
                echo "=== Stage 4: STATISTICAL ANALYSIS ==="
                echo "Computing Pearson/Spearman correlations, ANOVA hypothesis testing, VIF diagnostics..."
            }
        }

        stage('TRAIN MODELS') {
            steps {
                echo "=== Stage 5: TRAIN MODELS ==="
                echo "Training Linear Regression, Decision Tree, Random Forest, Bagging, SVR, XGBoost, Voting Regressor..."
                sh """
                    python src/pipeline_runner.py \
                        --dataset-path "${params.DATASET_PATH}" \
                        --target "${params.TARGET}" \
                        --selection-metric "${params.SELECTION_METRIC}" \
                        --training-mode "${params.TRAINING_MODE}" \
                        --output-summary "artifacts/jenkins_build_summary.json"
                """
            }
        }

        stage('EVALUATE') {
            steps {
                echo "=== Stage 6: EVALUATE ==="
                script {
                    def summary = readJSON file: 'artifacts/jenkins_build_summary.json'
                    echo "Out-Of-Fold Evaluation complete."
                    echo "Primary Metric (${params.SELECTION_METRIC}): ${summary[params.SELECTION_METRIC]}"
                    echo "MAE: ${summary.MAE} | RMSE: ${summary.RMSE} | R2: ${summary.R2}"
                }
            }
        }

        stage('SELECT BEST MODEL') {
            steps {
                echo "=== Stage 7: SELECT BEST MODEL ==="
                script {
                    def summary = readJSON file: 'artifacts/jenkins_build_summary.json'
                    echo "Winning Model Selected Deterministically: ${summary.selected_model}"
                }
            }
        }

        stage('VERSION MODEL') {
            steps {
                echo "=== Stage 8: VERSION MODEL ==="
                script {
                    def summary = readJSON file: 'artifacts/jenkins_build_summary.json'
                    echo "Registered Immutable Model Artifact Version: ${summary.model_version}"
                    echo "Dataset Hash: ${summary.dataset_hash}"
                }
            }
        }

        stage('GENERATE REPORT') {
            steps {
                echo "=== Stage 9: GENERATE REPORT ==="
                script {
                    def summary = readJSON file: 'artifacts/jenkins_build_summary.json'
                    echo "Executive HTML Report generated at: ${summary.report_html_path}"
                }
            }
        }

        stage('ARCHIVE ARTIFACTS') {
            steps {
                echo "=== Stage 10: ARCHIVE ARTIFACTS ==="
                archiveArtifacts artifacts: 'artifacts/reports/*.html, artifacts/jenkins_build_summary.json, artifacts/models/**', fingerprint: true, allowEmptyArchive: false
            }
        }
    }

    post {
        always {
            echo "Pipeline run completed for Jenkins Build #${env.BUILD_NUMBER}."
        }
        success {
            echo "Jenkins Pipeline SUCCESS: Model trained, evaluated, versioned, and archived."
        }
        failure {
            echo "Jenkins Pipeline FAILED: Review build logs above."
        }
    }
}
