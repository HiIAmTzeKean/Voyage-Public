# Production ML Academy: 3-Project Curriculum - Executive Overview

## 12-Week Program at a Glance

```
Week 1:     Onboarding & Foundation
            ↓
Week 2-3:   PROJECT 1: Time Series Analysis (Build)
            ↓
Week 4:     PROJECT 1: Time Series Analysis (Deploy)
            ↓
Week 5-6:   PROJECT 2: Agentic Systems (Build)
            ↓
Week 7-9:   PROJECT 2: Agentic Systems (Polish)
            ↓
Week 10:    PROJECT 3: Full-Stack ETL (Build)
            ↓
Week 11-12: PROJECT 3: Full-Stack ETL (Deploy)
```

---

## Project Overview Table

| Project | Weeks | Type | Business Value | Tech Stack | Deliverable |
|---------|-------|------|-----------------|-----------|------------|
| **1. Time Series Analysis** | 2-4 | Cohort-based analysis | Forecasting capabilities | Snowflake, Python (statsmodels, PyTorch), Streamlit | Deployed forecasting API + monitoring dashboard |
| **2. Agentic Systems** | 5-9 | Individual implementation | Autonomous analytics | MCP tools, LangChain, OpenAI/Claude, Streamlit | Deployed analytics agent + tool ecosystem |
| **3. Full-Stack ETL + Dashboard** | 10-12 | Individual implementation | Data infrastructure | Python async, dbt/Airflow, Snowflake, Streamlit | Live production dashboard + operational runbook |

---

## PROJECT 1: Time Series Analysis & Hypothesis Testing

### The Challenge
*"Analyze real-world time series data. Formulate testable hypotheses. Validate using statistical rigor and machine learning."*

### Key Learning Outcomes
- Hypothesis formulation and statistical testing (ADF test, Granger causality, cointegration)
- Time series fundamentals (stationarity, seasonality, decomposition)
- Model comparison (ARIMA/SARIMA vs. Prophet vs. LSTM)
- Snowflake for warehouse analytics
- Building production ML pipelines

### Phase Breakdown

**Phase A (Week 2): Problem Framing & EDA**
- Formulate 3+ testable hypotheses
- Exploratory data analysis with seasonal decomposition
- Baseline model establishment
- Deliverable: Problem statement + EDA notebook

**Phase B (Week 3): Statistical Testing & Modeling**
- Run statistical tests (ADF, Granger causality, cointegration)
- Compare multiple modeling approaches (baseline, ARIMA, Prophet, LSTM)
- Rigorous cross-validation and residual analysis
- Deliverable: Model comparison report with insights

**Phase C (Week 4): Production Pipeline**
- Modularize code (data loading → preprocessing → inference)
- Build REST API (FastAPI)
- Containerize (Docker)
- Deploy monitoring dashboard (Streamlit)
- Deliverable: Deployed API + monitoring system

### Tech Stack
```
Data: Snowflake (SQL) + Python
Models: statsmodels (ARIMA), Prophet, PyTorch (LSTM)
Visualization: Matplotlib, Plotly, Streamlit
Deployment: Docker, FastAPI
```

### Success Metrics
- Hypotheses: 3+ specific, testable claims ✓
- Statistical Rigor: All tests with p-values, no p-hacking ✓
- Model Accuracy: RMSE within 10% of best baseline ✓
- Code Quality: >60% test coverage, PEP 8 compliant ✓
- Deployment: API + dashboard functional ✓

---

## PROJECT 2: Agentic Systems with MCP & LangChain

### The Challenge
*"Build an autonomous agent that reasons about analytics problems, uses tools to gather data, and provides intelligent insights."*

### Key Learning Outcomes
- Model Context Protocol (MCP) standards and architecture
- LangChain agent patterns (ReAct, tool selection, state management)
- Designing production MCP tools (SQL, file access, API calls)
- Multi-step reasoning and chain-of-thought
- Cost tracking and observability for LLM applications
- Production error handling and recovery

### Phase Breakdown

**Phase A (Weeks 5-6): Tool Design & Architecture**
- Define 3+ MCP tools (SQL query, file browser, REST API, etc.)
- Implement MCP servers with input validation and security
- Design error handling and audit logging
- Deliverable: MCP tool specifications + working tool implementations

**Phase B (Weeks 6-7): Agent Workflows & Implementation**
- Build agent using LangChain + LangGraph
- Implement 3+ distinct workflows (investigate anomaly, compare datasets, forecast)
- Set up reasoning traces and multi-step planning
- Implement cost tracking and token monitoring
- Deliverable: Working agent with multiple workflows

**Phase C (Weeks 8-9): Integration & Production Readiness**
- End-to-end integration testing
- Build FastAPI endpoints for agent invocation
- Production monitoring dashboard (Streamlit)
- Deploy with error handling and observability
- Deliverable: Deployed agent system + comprehensive monitoring

### Tech Stack
```
Framework: LangChain + LangGraph
LLM: OpenAI GPT-4 or Anthropic Claude
MCP Tools: Custom Python servers
Data Access: Snowflake, REST APIs
Observability: LangSmith
Deployment: FastAPI + Docker
Monitoring: Streamlit dashboard
```

### Success Metrics
- MCP Tools: 3+ tools with validated schemas ✓
- Workflows: 3+ distinct multi-step workflows ✓
- Reasoning: Agent traces show chain-of-thought ✓
- Error Handling: Graceful failures, no crashes ✓
- Production: Logging, monitoring, cost tracking ✓

### Example Workflows
1. **Investigate Anomaly**: "Is 2025-01-25 an anomaly?" → Agent queries data, runs statistical tests, provides root cause analysis
2. **Compare Datasets**: "How does region A performance compare to region B?" → Agent aggregates, tests, visualizes, provides insights
3. **Build Forecast**: "Forecast next week's demand" → Agent retrieves data, calls model API, returns forecast with explanation

---

## PROJECT 3: Full-Stack ETL Pipeline + Streamlit Dashboard

### The Challenge
*"Build an end-to-end data platform that ingests from APIs, transforms data with optimal concurrency, and serves live analytics."*

### Key Learning Outcomes
- Concurrent API fetching (asyncio, rate limit handling, retries)
- ETL pipeline design and data quality frameworks
- dbt or Python-based transformations
- Data warehouse schema design and optimization
- Building production Streamlit dashboards
- Operational monitoring and alerting
- Documentation and runbooks for production systems

### Phase Breakdown

**Phase A (Week 10): ETL Architecture & API Integration**
- Analyze 2-3 data sources (APIs, databases)
- Design concurrent fetcher with rate limit handling
- Implement validation schema (Great Expectations)
- Error handling and retry strategy
- Deliverable: Working concurrent fetcher + data validation

**Phase B (Week 11): Transformation & Load**
- Design data warehouse schema (raw → transformed layers)
- Implement transformations (dbt SQL or Python Polars)
- Set up incremental loading strategy
- Build data quality framework
- Schedule with Airflow/Prefect
- Deliverable: Automated ETL pipeline, data quality reports

**Phase C (Week 12): Dashboard & Operations**
- Build interactive Streamlit dashboard
- Performance optimization (caching, query optimization)
- Implement monitoring and alerting
- Write operational runbook and architecture docs
- Deploy with Docker
- Deliverable: Live production dashboard + operational documentation

### Tech Stack
```
Ingestion: Python (aiohttp for async), requests
Orchestration: Airflow or Prefect
Transformation: dbt (SQL) or Polars (Python)
Warehouse: Snowflake or BigQuery
Dashboard: Streamlit
Monitoring: Python logging, Prometheus (optional)
Deployment: Docker, cloud platform
```

### Success Metrics
- Concurrent Fetching: 3+ APIs fetched in parallel ✓
- Data Quality: >95% records pass validation ✓
- Transformations: Clear raw → transformed pipeline ✓
- Dashboard: <3s page load time, interactive ✓
- Monitoring: Alerts fire on failures/staleness ✓
- Documentation: README + Architecture + Runbook ✓

### Pipeline Architecture
```
API Source 1 ─┐
API Source 2 ─┼→ Async Fetcher → Validation → Transform → Snowflake → Streamlit
API Source 3 ─┘                                    ↓        Dashboard
                                        Great Expectations
                                        Data Quality Tests
                                              ↓
                                         Monitoring
                                         Alerting
```

---

## Curriculum Progression

### Skill Arc Across 3 Projects

| Dimension | Project 1 | Project 2 | Project 3 |
|-----------|-----------|-----------|-----------|
| **Statistical Rigor** | Deep (hypothesis testing) | Medium (agent prompting) | Minimal |
| **Reasoning Complexity** | Linear (EDA → models) | Multi-step (agent planning) | Deterministic (ETL pipeline) |
| **Production Focus** | API + monitoring | Observability + cost tracking | Operations + maintenance |
| **Scale** | Medium (single model) | Medium (agentic workflows) | Large (data pipeline) |

### Portfolio Value

**After Week 4**: Completed time series forecasting system
- Demonstrates: Statistical thinking, ML modeling, deployment skills
- Recruiter view: "Can they analyze data rigorously and ship systems?"

**After Week 9**: Deployed agentic analytics system
- Demonstrates: Advanced ML, reasoning with LLMs, production LLM applications
- Recruiter view: "Can they build with cutting-edge AI and reason about tool design?"

**After Week 12**: Full-stack data platform
- Demonstrates: Systems architecture, data engineering, operational excellence
- Recruiter view: "Can they build reliable, maintainable data infrastructure?"

Together: **3 portfolio projects showcasing analytical depth + systems thinking + production maturity**

---

## Weekly Meeting Structure

**60-minute weekly 1:1s** (consistent cadence essential):
- 0-5 min: Check-in & rapport
- 5-15 min: Progress review ("What did you accomplish this week?")
- 15-40 min: Technical deep-dive (code review, architecture decisions, debugging)
- 40-50 min: Guidance & unblocking ("Here's direction for next week")
- 50-60 min: Career skills (resume, interview prep, communication)

**Mentee Preparation**: Pushed code, specific questions, brief progress update
**Mentor Preparation**: Code review before meeting, 1-2 targeted feedback items

---

## GitHub Portfolio End-State

After 12 weeks, each mentee will have:

**Repository 1: Time Series Forecasting System**
- Professional README with problem context, approach, results
- Modular source code (data → preprocessing → models → inference)
- Unit tests (>60% coverage)
- Docker setup for reproduction
- Example predictions + metrics
- Blog post (optional): "How I Built a Time Series Forecasting System"

**Repository 2: Analytics Agent**
- Architecture documentation explaining MCP + LangChain design
- MCP tool implementations with security specs
- Example agent workflows (with transcripts)
- Cost tracking and monitoring dashboard setup
- Production deployment guide
- Blog post (optional): "Building Agentic Systems with LangChain & MCP"

**Repository 3: ETL Platform**
- Complete runbook for operations
- Data schema documentation
- Transformation logic (dbt models or Python code)
- Live dashboard link / deployment instructions
- Monitoring setup + alerts configuration
- Performance benchmarks (throughput, latency)
- Blog post (optional): "Building a Production ETL Pipeline"

**Resume Impact**:
```
Data Science & ML Experience
• Designed and deployed time series forecasting system using LSTM and ARIMA, 
  achieving RMSE within 10% of best baseline on Snowflake data
• Built autonomous analytics agent with 5+ MCP tools, LangChain orchestration, 
  and LLM reasoning; processed 1000+ queries with <3% error rate
• Engineered end-to-end ETL pipeline ingesting from 3+ APIs concurrently, 
  with dbt transformations and Streamlit dashboard serving real-time analytics; 
  reduced data latency from 24h to 2h

Key Skills: Time Series Analysis • ML Deployment • Agentic AI • Data Engineering • 
Snowflake • Python • LangChain • dbt • ETL • Streamlit • FastAPI • Docker
```

---

## Assessment Rubric (3-Point Scale)

### Passing (3/5): Masters-level, ready to work
- Technical approach is sound and justified
- Code is modular, tested, and documented
- System is deployed and monitored
- Communication is clear

### Strong (4-5/5): Recruiter-worthy
- Demonstrates mastery and production maturity
- Innovative problem-solving
- Excellent documentation and communication
- Operations and monitoring are thoughtful

---

## Next Steps

1. **Communicate Program Structure** to mentees (Week 1)
2. **Conduct Baseline Assessment** (Week 1): strengths, gaps, career goals
3. **Select Project Themes** (Week 1): time series dataset, agent use case, ETL data sources
4. **Set Up Environments** (Week 1): Git repos, Snowflake access, API keys
5. **Establish Meeting Cadence** (Week 1): recurring weekly 1:1s
6. **Kick Off Project 1** (Week 2): hypothesis formation, EDA

---

**Document**: 3-Project Curriculum Overview  
**Version**: 1.0  
**Last Updated**: January 2026

For detailed specifications, phase-by-phase guidance, and assessment rubrics, see **3_project_curriculum.md**
