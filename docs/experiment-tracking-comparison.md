# Self-Hosted Experiment Tracking: Comparison Guide

This document compares self-hosted experiment tracking solutions for tracking Hydra CLI experiments in the align-system.

## Executive Summary

| Solution | Hydra Integration | Setup Complexity | Infrastructure | Best For |
|----------|-------------------|------------------|----------------|----------|
| **ClearML** | ⭐⭐⭐ Native | Medium | Docker/K8s | Full MLOps + best Hydra support |
| **Aim** | ⭐⭐⭐ Native | Low | Single process | Modern UI + easy setup |
| **MLflow** | ⭐⭐ Manual | Medium | Docker + DB | ML lifecycle + model registry |
| **SQLite** | ⭐⭐ Manual | None | Local file | Zero deps + portable |

**Recommendation**: For self-hosted with Hydra, **ClearML** or **Aim** offer the best integration with native OmegaConf support. For minimal overhead, **SQLite** requires zero infrastructure.

---

## Detailed Comparison

### 1. ClearML

#### Overview
ClearML is a full-stack, open-source MLOps platform with **native Hydra/OmegaConf support**. It automatically captures OmegaConf configurations without any manual conversion.

#### Hydra Integration Quality: ⭐⭐⭐ Excellent

```python
from clearml import Task
import hydra
from omegaconf import DictConfig

@hydra.main(config_path="conf", config_name="config")
def train(cfg: DictConfig):
    # ClearML automatically logs the OmegaConf config!
    task = Task.init(project_name="align-experiments", task_name="training")
    # cfg is automatically captured in CONFIGURATION > HYPERPARAMETERS > HYDRA

    for epoch in range(cfg.training.epochs):
        task.get_logger().report_scalar("loss", "train", value=loss, iteration=epoch)
```

No manual `OmegaConf.to_container()` conversion needed - ClearML handles it natively.

#### Deployment

**Docker Compose (Recommended):**
```yaml
version: "3.8"
services:
  clearml-server:
    image: allegroai/clearml:latest
    ports:
      - "8080:8080"   # Web UI
      - "8008:8008"   # API
      - "8081:8081"   # File server
    volumes:
      - clearml-data:/opt/clearml/data
      - clearml-logs:/opt/clearml/logs

volumes:
  clearml-data:
  clearml-logs:
```

**Resource Requirements:**
- Minimum: 4GB RAM, 2 CPU cores, 50GB storage
- Recommended: 8GB RAM, 4 CPU cores, 200GB storage
- Scales with Kubernetes for teams

#### Features

| Feature | Supported |
|---------|-----------|
| Experiment tracking | ✅ |
| Hydra config capture | ✅ Native |
| Model versioning | ✅ |
| Pipeline orchestration | ✅ |
| Hyperparameter tuning | ✅ |
| Dataset versioning | ✅ |
| GPU monitoring | ✅ |
| Multi-user | ✅ Unlimited |
| REST API | ✅ |
| Python SDK | ✅ |

#### Pros
- **Best-in-class Hydra integration** (zero configuration)
- Unlimited users on self-hosted (no per-seat fees)
- Full MLOps platform (not just tracking)
- Automatic framework detection (PyTorch, TensorFlow, etc.)
- Built-in pipeline automation
- Strong documentation

#### Cons
- Heavier infrastructure than simpler tools
- Learning curve for full platform features
- Requires Docker/Kubernetes knowledge for production
- UI can be overwhelming for simple use cases

#### Cost
- **Self-hosted: $0** (unlimited users, unlimited experiments)
- Only pay for your own infrastructure

---

### 2. Aim

#### Overview
Aim is a modern, open-source experiment tracking tool designed for performance at scale. It has **native OmegaConf support** and a beautiful, responsive UI that handles 100,000+ runs efficiently.

#### Hydra Integration Quality: ⭐⭐⭐ Excellent

```python
from aim import Run
import hydra
from omegaconf import DictConfig

@hydra.main(config_path="conf", config_name="config")
def train(cfg: DictConfig):
    run = Run(experiment="align-experiments")

    # Native OmegaConf support - assign directly!
    run["hparams"] = cfg  # No conversion needed

    for epoch in range(cfg.training.epochs):
        run.track(loss, name="loss", step=epoch, context={"subset": "train"})
```

#### Deployment

**Local (Simplest):**
```bash
# Initialize in your repo
aim init

# Start UI server
aim up  # Accessible at http://localhost:43800
```

**Remote Tracking Server (for teams):**
```bash
# On server
aim server --host 0.0.0.0 --port 53800

# From clients
run = Run(repo="aim://server-ip:53800")
```

**Docker:**
```bash
docker run -d \
    -p 43800:43800 \
    -p 53800:53800 \
    -v /path/to/aim:/aim \
    aimstack/aim
```

**Resource Requirements:**
- Minimum: 2GB RAM, 1 CPU core
- Scales with data volume (100k runs ≈ 10GB)
- Single process, no database required

#### Features

| Feature | Supported |
|---------|-----------|
| Experiment tracking | ✅ |
| Hydra config capture | ✅ Native |
| Model versioning | ❌ |
| Pipeline orchestration | ❌ |
| Hyperparameter tuning | ❌ |
| Dataset versioning | ❌ |
| GPU monitoring | ✅ |
| Multi-user | ✅ (via remote server) |
| REST API | ✅ |
| Python SDK | ✅ |
| SQL-like queries | ✅ |

#### Query API (Unique Feature)

```python
from aim import Repo

repo = Repo("/path/to/aim")

# Query runs with SQL-like syntax
for run in repo.query_runs("run.hparams.lr > 0.01 and run.hparams.batch_size == 32"):
    print(run.name, run["hparams"])

# Query metrics
for metric_collection in repo.query_metrics("metric.name == 'loss'"):
    steps, values = metric_collection.values.sparse_numpy()
```

#### Pros
- **Simplest self-hosted setup** (single command)
- Native OmegaConf support (no conversion)
- Performant UI (handles 100k+ runs)
- Powerful query API for analysis
- Lightweight resource requirements
- Active development

#### Cons
- Focused on tracking only (not full MLOps)
- No model registry
- Smaller community than MLflow
- No pipeline orchestration

#### Cost
- **Self-hosted: $0** (fully open source)

---

### 3. MLflow

#### Overview
MLflow is the most widely adopted open-source ML lifecycle platform. It requires **manual Hydra config conversion** but offers comprehensive features including model registry and deployment.

#### Hydra Integration Quality: ⭐⭐ Manual

```python
import mlflow
import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(config_path="conf", config_name="config")
def train(cfg: DictConfig):
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("align-experiments")

    with mlflow.start_run():
        # Manual conversion required!
        config_dict = OmegaConf.to_container(cfg, resolve=True)

        # Flatten nested config (MLflow params must be flat)
        def flatten(d, parent_key=""):
            items = []
            for k, v in d.items():
                new_key = f"{parent_key}.{k}" if parent_key else k
                if isinstance(v, dict):
                    items.extend(flatten(v, new_key).items())
                else:
                    items.append((new_key, str(v)[:500]))  # 500 char limit
            return dict(items)

        mlflow.log_params(flatten(config_dict))

        for epoch in range(cfg.training.epochs):
            mlflow.log_metric("loss", loss, step=epoch)
```

#### Deployment

**Docker Compose (Production):**
```yaml
version: "3.8"
services:
  mlflow:
    image: ghcr.io/mlflow/mlflow:latest
    ports:
      - "5000:5000"
    environment:
      - BACKEND_STORE_URI=postgresql://mlflow:password@postgres:5432/mlflow
      - ARTIFACT_ROOT=s3://mlflow-artifacts
    command: >
      mlflow server
      --backend-store-uri postgresql://mlflow:password@postgres:5432/mlflow
      --default-artifact-root s3://mlflow-artifacts
      --host 0.0.0.0

  postgres:
    image: postgres:15
    environment:
      - POSTGRES_USER=mlflow
      - POSTGRES_PASSWORD=password
      - POSTGRES_DB=mlflow
    volumes:
      - postgres-data:/var/lib/postgresql/data

volumes:
  postgres-data:
```

**Simple Local (SQLite):**
```bash
mlflow server \
    --backend-store-uri sqlite:///mlflow.db \
    --default-artifact-root ./mlflow-artifacts \
    --host 127.0.0.1 \
    --port 5000
```

**Resource Requirements:**
- Minimum: 2GB RAM, 1 CPU core (SQLite)
- Production: 4GB RAM, 2 CPU cores + PostgreSQL + S3

#### Features

| Feature | Supported |
|---------|-----------|
| Experiment tracking | ✅ |
| Hydra config capture | ⚠️ Manual |
| Model versioning | ✅ Model Registry |
| Pipeline orchestration | ⚠️ Basic (Projects) |
| Hyperparameter tuning | ❌ (use Optuna) |
| Dataset versioning | ❌ |
| GPU monitoring | ❌ |
| Multi-user | ✅ |
| REST API | ✅ |
| Python/R/Java SDK | ✅ |
| Model Serving | ✅ |

#### Pros
- Largest community and ecosystem
- Model registry with staging/production
- Multi-language support (Python, R, Java)
- Built-in model serving
- Extensive documentation
- Auto-logging for many frameworks

#### Cons
- **Manual Hydra config conversion required**
- More complex production setup (needs DB + object storage)
- Basic UI compared to newer tools
- No native OmegaConf support
- Parameter string length limits (500 chars)

#### Cost
- **Self-hosted: $0** (open source)
- Infrastructure costs for PostgreSQL + S3

---

### 4. SQLite (DIY)

#### Overview
A minimal, zero-dependency solution using Python's built-in `sqlite3` module. Perfect for local experiments where you want tracking without any infrastructure overhead.

#### Hydra Integration Quality: ⭐⭐ Manual (but simple)

```python
from align_track import SQLiteTracker

tracker = SQLiteTracker("experiments.db")

@hydra.main(config_path="conf", config_name="config")
def train(cfg):
    with tracker.start_run("align-experiments", config=cfg) as run:
        # Config is automatically flattened and stored
        for epoch in range(cfg.training.epochs):
            run.log({"loss": loss, "accuracy": acc})
        run.log_artifact("model.pt")
```

#### Deployment

**No deployment needed!** Just use the Python module:

```python
from align_track import track_run

with track_run("my-experiment", config=hydra_cfg) as run:
    run.log({"metric": value})
```

The database is a single portable file.

**Resource Requirements:**
- Minimum: None (uses local disk)
- Storage: ~1KB per run + metrics

#### Features

| Feature | Supported |
|---------|-----------|
| Experiment tracking | ✅ |
| Hydra config capture | ⚠️ Auto-flatten |
| Model versioning | ❌ |
| Pipeline orchestration | ❌ |
| Hyperparameter tuning | ❌ |
| Dataset versioning | ❌ |
| GPU monitoring | ❌ |
| Multi-user | ❌ (file locking) |
| REST API | ❌ |
| Python SDK | ✅ |
| CLI queries | ✅ |
| Raw SQL | ✅ |

#### CLI Usage

```bash
# List experiments
experiments list

# Show run details
experiments show 42 --metrics

# Compare runs
experiments compare 42 43 44

# Custom SQL
experiments query "SELECT key, AVG(value) FROM metrics GROUP BY key"
```

#### Pros
- **Zero dependencies** (built-in sqlite3)
- **Zero infrastructure** (single file)
- Portable and git-friendly
- Full SQL query power
- Fastest setup possible
- Works offline

#### Cons
- No web UI
- Single-user only (file locking issues with concurrent writes)
- Manual visualization (export to pandas/matplotlib)
- No collaboration features
- Basic compared to full platforms

#### Cost
- **$0** (zero dependencies, zero infrastructure)

---

## Decision Matrix

### By Team Size

| Team Size | Recommended | Reason |
|-----------|-------------|--------|
| Solo developer | SQLite or Aim | Minimal overhead |
| Small team (2-5) | Aim or ClearML | Easy setup, good collaboration |
| Medium team (5-20) | ClearML | Full MLOps, unlimited users |
| Large team (20+) | ClearML or MLflow | Enterprise features, scalability |

### By Use Case

| Use Case | Recommended | Reason |
|----------|-------------|--------|
| Quick prototyping | SQLite | Zero setup |
| Hydra experiments | ClearML or Aim | Native OmegaConf |
| Model deployment | MLflow | Model registry + serving |
| Full MLOps pipeline | ClearML | Pipelines + orchestration |
| Maximum simplicity | SQLite | No dependencies |
| Best UI/UX | Aim | Modern, performant |

### By Infrastructure

| Infrastructure | Recommended |
|----------------|-------------|
| No infrastructure | SQLite |
| Single server | Aim |
| Docker available | ClearML or MLflow |
| Kubernetes | ClearML |
| Cloud storage (S3/GCS) | MLflow |

---

## Quick Start Guides

### ClearML Quick Start

```bash
# 1. Start server
docker run -d -p 8080:8080 -p 8008:8008 -p 8081:8081 allegroai/clearml:latest

# 2. Configure client
pip install clearml
clearml-init  # Enter server URL: http://localhost:8080

# 3. Use in code
from clearml import Task
task = Task.init(project_name="align", task_name="exp1")
# Hydra config automatically captured!
```

### Aim Quick Start

```bash
# 1. Install
pip install aim

# 2. Initialize
aim init

# 3. Use in code
from aim import Run
run = Run()
run["hparams"] = hydra_cfg  # Native OmegaConf!
run.track(loss, name="loss", step=epoch)

# 4. View UI
aim up  # http://localhost:43800
```

### MLflow Quick Start

```bash
# 1. Install
pip install mlflow

# 2. Start server
mlflow server --host 127.0.0.1 --port 5000

# 3. Use in code
import mlflow
mlflow.set_tracking_uri("http://localhost:5000")
with mlflow.start_run():
    mlflow.log_params(OmegaConf.to_container(cfg, resolve=True))
    mlflow.log_metric("loss", loss)

# View UI at http://localhost:5000
```

### SQLite Quick Start

```python
# No setup needed!
from align_track import track_run

with track_run("my-experiment", config=hydra_cfg) as run:
    run.log({"loss": 0.5})
    run.log_artifact("model.pt")

# Query via CLI
# experiments list
# experiments show 1 --metrics
```

---

## Recommendation for align-system

Given the requirements for tracking Hydra CLI experiments in the itm-kitware/align-system:

### Primary Recommendation: **ClearML**

1. **Native Hydra/OmegaConf support** - Zero configuration needed
2. **Unlimited self-hosted users** - No per-seat costs
3. **Full MLOps capabilities** - Grows with your needs
4. **Strong pipeline support** - Important for complex ML workflows

### Alternative: **Aim**

If you want simpler setup and don't need full MLOps:
1. Single-command setup
2. Native OmegaConf support
3. Excellent query API for analysis
4. Modern, performant UI

### Fallback: **SQLite**

For quick experiments without infrastructure:
1. Zero dependencies
2. Works immediately
3. Portable single file
4. Full SQL queries

---

## Implementation in align-track

All approaches have been implemented in separate branches:

```bash
# ClearML
git checkout claude/clearml-tracking-dcAPk
pip install align-track[clearml]

# Aim
git checkout claude/aim-tracking-dcAPk
pip install align-track[aim]

# MLflow
git checkout claude/mlflow-tracking-dcAPk
pip install align-track[mlflow]

# SQLite (default, no extra deps)
git checkout claude/sqlite-tracking-dcAPk
pip install align-track
```

Choose the approach that best fits your team's infrastructure and requirements.
