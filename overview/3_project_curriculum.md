# Production ML Academy: 3-Project Curriculum (12-Week Intensive)

## Modified Program Timeline

| Phase | Weeks | Project Focus | Core Objectives | Deliverable |
|-------|-------|---------------|-----------------|-------------|
| **Onboarding** | 1 | Foundation setup | Baseline assessment, environment setup, team dynamics | Development plan + tech stack validated |
| **Project 1: Build** | 2-3 | **Time Series Analysis** | EDA on time series data, hypothesis formulation, stationarity testing | Problem statement + hypothesis doc + baseline models |
| **Project 1: Deploy** | 4 | **Time Series (TS)** | Modular ML pipeline in Python, containerization, Snowflake integration | Deployed TS models + monitoring |
| **Project 2: Build** | 5-6 | **Agentic Systems** | MCP tool design, LangChain workflows, analytics agent development | MCP tool specifications + agent workflows |
| **Project 2: Polish** | 7-9 | **Agentic Systems** | Integration testing, error handling, documentation | Deployed agentic system + GitHub repo |
| **Project 3: Build** | 10 | **Full-Stack ETL** | API design, ETL architecture, concurrency patterns | ETL pipeline code + data schema |
| **Project 3: Deploy** | 11-12 | **Full-Stack ETL** | End-to-end ingestion, Streamlit dashboard, monitoring & maintenance | Live Streamlit dashboard + operational docs |

---

## PROJECT 1: Time Series Analysis & Hypothesis Testing (Weeks 2-4)

### Project Overview

Students will work **as a cohort** (not individual projects) to analyze a real-world time series dataset using production tools. They'll simulate a corporate analytics environment by using a data warehouse (Snowflake), formulate testable hypotheses about the data, and validate them using statistical tests and ML models.

### Business Context

Time series analysis is foundational for industries spanning finance (price prediction), energy (demand forecasting), e-commerce (seasonal trends), and SaaS (user growth). This project teaches how to think about temporal data rigorously and which statistical/ML approaches apply to which problems.

### Learning Outcomes

- **Time Series Fundamentals**: Stationarity (ADF test), seasonality detection, trend decomposition
- **Hypothesis-Driven Analysis**: Formulate testable hypotheses (e.g., "energy demand has weekly seasonality" or "sales trends differ by region")
- **Statistical Testing**: Granger causality, cointegration, correlation significance
- **ML Modeling**: ARIMA/SARIMA, Prophet, LSTM comparisons with cross-validation
- **Data Warehouse Skills**: SQL window functions, time-based aggregations in Snowflake
- **Visualization & Communication**: Tableau/Matplotlib for time series patterns, presenting findings

### Tech Stack

| Component | Tools | Purpose |
|-----------|-------|---------|
| **Data Warehouse** | Snowflake | Central analytics hub; time series functions (RESAMPLE, INTERPOLATE) |
| **Analysis & ML** | Python (pandas, statsmodels, scikit-learn, Prophet, PyTorch) | Statistical tests, baseline + advanced models |
| **Visualization** | Tableau, Matplotlib, Plotly | Dashboards, trend plots, hypothesis validation |
| **Version Control** | Git + GitHub | Notebook versioning, code collaboration |
| **Orchestration** | Airflow (optional) | Schedule retraining pipelines |

### Phase A: Problem Framing & EDA (Week 2)

**Mentor Checkpoint**: Review problem statement for clarity; validate hypothesis testability

**Deliverables**:
1. **Time Series Problem Statement** (2-3 pages)
   - Dataset description: source, time range, granularity, missing data
   - Initial observations: plots, summary statistics
   - Hypothesis 1, 2, 3: specific, testable claims (e.g., "Daily energy demand has a 7-day seasonality")
   - Baseline metrics: benchmark models to beat (e.g., naive forecast, simple exponential smoothing)

2. **EDA Notebook** with:
   - Time series plots (raw data + rolling averages)
   - Seasonal decomposition (trend, seasonal, residual)
   - Missing data analysis and handling strategy
   - Correlation heatmaps (internal relationships)
   - Autocorrelation (ACF) and partial autocorrelation (PACF) plots

3. **Snowflake Schema Design**:
   - Raw data table (with timestamp PK)
   - Aggregated/processed table (hourly/daily granularity)
   - Sample SQL queries demonstrating time-based window functions

**Example Hypotheses**:
- "Energy demand exhibits strong weekly seasonality (7-day cycle)"
- "Weather variables (temperature, humidity) Granger-cause energy demand"
- "Energy demand shows a structural break (level shift) post-COVID"
- "Sales across regions are cointegrated; they move together long-term"

### Phase B: Hypothesis Testing & Modeling (Week 3)

**Mentor Checkpoint**: Validate experimental rigor; review statistical test selection; ensure no p-hacking

**Deliverables**:
1. **Statistical Testing Report**:
   - ADF test results: Is the series stationary? (reject null = stationary)
   - Granger causality: Does feature X help forecast target Y?
   - Cointegration test: Do two series move together long-term?
   - Correlation significance: Are relationships statistically meaningful?
   - All with p-values, interpretation, and implications for modeling

2. **Model Comparison Notebook**:
   - **Baseline**: Naive forecast (repeat last value), seasonal naive, simple exponential smoothing
   - **Traditional**: ARIMA/SARIMA with grid search (p, d, q selection via ACF/PACF), Prophet
   - **Modern**: LSTM (1-2 layer), TCN (Temporal Convolutional Network)
   - **Ensemble**: Voting average of 2-3 models
   - **Metrics**:
     - MAE, RMSE, MAPE (accuracy)
     - Directional accuracy (did model predict correct trend?)
     - Performance by season/region (segment analysis)
   - **Learning curves**: Train/validation split plots showing overfitting risk
   - **Residual analysis**: Are residuals white noise? (Ljung-Box test)

3. **Model Selection Justification**:
   - Why did you choose ARIMA over Prophet or vice versa?
   - Which model performed best and why?
   - Trade-offs: interpretability vs. accuracy, training time vs. performance

### Phase C: Production Pipeline (Week 4)

**Mentor Checkpoint**: Code review; Docker build test; API functionality validation

**Deliverables**:
1. **Modular Python Code** with clear separation:
   ```
   src/
   ├── data_loader.py          # Snowflake connection, data ingestion
   ├── preprocessing.py         # Stationarity transformation, scaling
   ├── models/
   │   ├── arima_model.py
   │   ├── prophet_model.py
   │   ├── lstm_model.py
   │   └── ensemble.py
   ├── inference.py             # Batch prediction logic
   ├── evaluation.py            # Metrics computation
   ├── config.yaml              # Hyperparameters, Snowflake credentials
   └── utils/
       ├── logging.py           # Logging setup
       └── validation.py        # Data quality checks
   ```

2. **Testing**:
   - Unit tests for data loading, preprocessing, model training
   - Integration test: full pipeline from Snowflake → prediction
   - Target: >60% code coverage
   - `pytest tests/` passes without warnings

3. **Dockerization**:
   ```dockerfile
   FROM python:3.11-slim
   WORKDIR /app
   COPY requirements.txt .
   RUN pip install -r requirements.txt
   COPY src/ src/
   COPY config.yaml .
   CMD ["python", "-m", "src.inference"]
   ```
   - `docker build -t ts-model .` succeeds
   - Container runs inference on sample data

4. **API Endpoint** (FastAPI):
   - `POST /predict` - Accept JSON with time series features, return forecast
   - Input validation: date range, feature completeness
   - Error handling: graceful failure for edge cases
   - Response includes: forecast values, confidence intervals, model metadata

5. **Monitoring Dashboard** (Streamlit):
   - Real-time forecast vs. actuals plot
   - Model performance metrics (MAE, RMSE) over time
   - Data quality flags: missing values, anomalies
   - Alerts: model performance degradation (trigger retraining)

**Example API Request/Response**:
```json
POST /predict
{
  "start_date": "2025-01-01",
  "end_date": "2025-02-01",
  "model": "ensemble",
  "include_confidence": true
}

Response:
{
  "forecast": [23.5, 24.1, 22.8, ...],
  "confidence_lower": [20.1, 21.3, 19.9, ...],
  "confidence_upper": [26.9, 26.9, 25.7, ...],
  "model_used": "ensemble",
  "rmse": 2.34,
  "mae": 1.89,
  "last_updated": "2025-01-28T11:33:00Z"
}
```

### Success Criteria

| Criterion | Threshold | Assessment |
|-----------|-----------|------------|
| Hypothesis formulation | 3+ specific, testable hypotheses | Clear and falsifiable |
| Statistical rigor | All tests have p-values reported | No p-hacking (multiple testing correction applied) |
| Model accuracy | RMSE within 10% of best baseline | Achieves reasonable forecast quality |
| Code quality | >60% test coverage, PEP 8 compliant | Modular, documented, reproducible |
| Deployment | API + dashboard functional | Can serve real-time predictions |
| Documentation | README + problem statement | Newcomer can run pipeline in <10 min |

### Recommended Dataset Options

- **Energy Domain** (your background): Hourly electricity demand/pricing, weather data
- **Finance**: Stock prices, trading volume, correlations
- **E-commerce**: Daily sales by product/region, traffic patterns
- **IoT/Sensors**: Temperature, humidity, sensor failures over time

---

## PROJECT 2: Agentic Systems with MCP & LangChain (Weeks 5-9)

### Project Overview

Students will build an **autonomous agentic system** using Model Context Protocol (MCP) tools and LangChain workflows. The agent will reason about analytics problems, call MCP tools to gather data, execute queries, and provide intelligent recommendations. This mirrors production AI systems used in companies like Anthropic, OpenAI, and major tech platforms.

### Business Context

Agentic systems represent the cutting edge of AI applications: they combine LLMs with tool use, reasoning, and autonomous decision-making. Companies are rapidly building agents for customer support, data analysis, content creation, and complex workflows. This project teaches both the technical architecture and the strategic thinking behind autonomous systems.

### Learning Outcomes

- **MCP Standards & Architecture**: Tool protocols, schema definitions, security model
- **LangChain Agent Patterns**: ReAct (Reasoning + Acting), tool selection, state management with LangGraph
- **Tool Design**: Building MCP servers for analytics (SQL queries, file access, API calls)
- **Agent Reasoning**: Prompting techniques, chain-of-thought, tool use planning
- **Workflow Orchestration**: Multi-step reasoning, error recovery, validation
- **Production Concerns**: Logging, monitoring, cost tracking (LLM calls), security

### Tech Stack

| Component | Tools | Purpose |
|-----------|-------|---------|
| **Agent Framework** | LangChain + LangGraph | Chain composition, agent orchestration, memory |
| **MCP Tools** | Custom MCP servers, Anthropic SDK | Tool standardization, schema definitions |
| **LLM** | OpenAI (GPT-4), Anthropic Claude, or open-source (Llama) | Reasoning engine |
| **Data Access** | Snowflake, DuckDB, REST APIs | MCP tool implementations |
| **Observability** | LangSmith | Trace debugging, monitoring |
| **Deployment** | FastAPI + Docker | Agent API service |

### Phase A: Tool Design & Architecture (Week 5)

**Mentor Checkpoint**: Review MCP tool schemas; validate tool use cases; ensure security model

**Deliverables**:
1. **MCP Tool Specifications** (document):
   - Tool 1: SQL Query Tool
     - Inputs: SQL query, database connection params
     - Outputs: Query result (rows, errors, execution time)
     - Safety guards: Query validation, timeout, rate limiting
   - Tool 2: File Browser Tool
     - Inputs: path, file pattern
     - Outputs: file contents or directory listing
   - Tool 3: REST API Caller
     - Inputs: URL, method, headers, params
     - Outputs: JSON response or error
   - Additional tools as needed (Slack notification, email, data validation, etc.)

2. **MCP Server Implementation** (code):
   ```python
   # mcp_servers/sql_tool.py
   from mcp.server import Server
   from mcp.types import Tool, TextContent, ToolInput
   
   server = Server("sql-analytics")
   
   @server.tool()
   def query_database(sql: str, max_rows: int = 1000) -> str:
       """Execute SQL query safely"""
       # Input validation: block DROP, DELETE
       # Execute query with timeout
       # Return formatted results
       pass
   
   @server.tool()
   def analyze_data(table: str, metric: str) -> dict:
       """Generate summary statistics"""
       pass
   ```

3. **Security & Compliance Document**:
   - Input validation: what queries are allowed? (no DROP, DELETE)
   - Rate limiting: max queries/minute per user
   - Audit logging: all tool calls logged with user, timestamp, parameters
   - Error handling: avoid leaking sensitive data in error messages

4. **Agent Architecture Diagram**:
   - LLM (brain) → LangGraph state machine → Tool selection → Tool execution → Response
   - Show loops: reasoning iteration, error recovery, output validation

### Phase B: Agent Workflows & Implementation (Weeks 6-7)

**Mentor Checkpoint**: Agent reasoning examples; test multi-step workflows; validate tool integration

**Deliverables**:
1. **Agent Implementation** (code):
   ```python
   # src/agent.py
   from langchain.agents import AgentExecutor, create_tool_calling_agent
   from langchain_core.prompts import ChatPromptTemplate
   from langchain_openai import ChatOpenAI
   
   llm = ChatOpenAI(model="gpt-4", temperature=0)
   
   # Tool list
   tools = [
       sql_query_tool,
       data_validation_tool,
       slack_notify_tool
   ]
   
   # Prompt with instructions for tool use
   prompt = ChatPromptTemplate.from_messages([
       ("system", "You are an expert data analyst. Use tools to investigate data questions..."),
       ("human", "{input}"),
       ("placeholder", "{agent_scratchpad}")
   ])
   
   agent = create_tool_calling_agent(llm, tools, prompt)
   executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
   
   # Example invocation
   result = executor.invoke({
       "input": "Is our energy demand normally distributed? Show me the distribution."
   })
   ```

2. **Workflow Examples** (test scenarios):
   - **Workflow 1**: "Investigate anomaly"
     - Agent queries: What happened on 2025-01-25?
     - Tool use: SQL to fetch data, statistical tests for outliers
     - Output: Anomaly report with root cause hypothesis
   
   - **Workflow 2**: "Compare datasets"
     - Agent queries: How does region A performance compare to region B?
     - Tool use: SQL aggregations for both, statistical tests, visualization
     - Output: Side-by-side comparison with insights
   
   - **Workflow 3**: "Build forecasting model"
     - Agent queries: Forecast next week's energy demand
     - Tool use: Data retrieval, feature engineering, model training (or calling existing model API)
     - Output: Forecast with confidence intervals and methodology explanation

3. **LangGraph State Machine** (if multi-agent):
   ```python
   from langgraph.graph import StateGraph
   
   workflow = StateGraph(AgentState)
   workflow.add_node("analyst", analyze_data_node)
   workflow.add_node("validator", validate_results_node)
   workflow.add_node("reporter", format_report_node)
   
   workflow.add_edge("analyst", "validator")
   workflow.add_edge("validator", "reporter")
   workflow.set_entry_point("analyst")
   
   app = workflow.compile()
   ```

4. **Error Handling & Recovery**:
   - Tool failure: SQL timeout → fallback to cached results, retry with limit
   - Invalid tool call: Model hallucinates parameter → constraint enforcement
   - Max iterations: Agent stuck in loop → graceful exit with partial results

5. **Cost & Token Tracking**:
   - Log every LLM API call: model, tokens in/out, cost
   - Dashboard showing total cost, cost per workflow, cost per user
   - Alerts for cost spikes or runaway loops

### Phase C: Integration, Testing & Deployment (Weeks 8-9)

**Mentor Checkpoint**: End-to-end workflow test; production readiness review; error scenarios

**Deliverables**:
1. **Integration Testing**:
   ```python
   def test_agent_workflow_investigate_anomaly():
       executor = setup_executor()
       result = executor.invoke({
           "input": "Is 2025-01-25 an anomaly?"
       })
       assert "anomaly" in result.lower() or "normal" in result.lower()
       assert result.tool_calls > 1  # Agent used multiple tools
   ```

2. **FastAPI Endpoint**:
   ```python
   @app.post("/ask")
   async def ask_agent(query: str, context: dict = None):
       """Invoke agent with a natural language query"""
       result = executor.invoke({
           "input": query,
           "context": context
       })
       return {
           "answer": result,
           "tools_used": result.tool_calls,
           "reasoning": result.thoughts,
           "cost": result.token_cost
       }
   ```

3. **Production Code Structure**:
   ```
   src/
   ├── mcp_servers/
   │   ├── sql_server.py
   │   ├── file_server.py
   │   └── api_server.py
   ├── agent/
   │   ├── executor.py        # Agent setup & orchestration
   │   ├── prompts.py         # System prompts & few-shot examples
   │   ├── tools.py           # Tool definitions
   │   └── workflows/
   │       ├── investigate.py
   │       ├── compare.py
   │       └── forecast.py
   ├── api/
   │   ├── main.py            # FastAPI app
   │   ├── schemas.py         # Request/response models
   │   └── middleware/
   │       ├── auth.py        # API key validation
   │       ├── logging.py     # Request/response logging
   │       └── cost_tracking.py
   └── monitoring/
       ├── traces.py          # LangSmith integration
       └── dashboards.py      # Streamlit monitoring UI
   ```

4. **Monitoring Dashboard** (Streamlit):
   - Agent call history: query, response, tools used
   - Success rate: % of queries resolved vs. failed
   - Cost tracking: total spend, cost per query, trending
   - Performance: response time distribution, token usage
   - Error logs: failures and recovery patterns

5. **GitHub Repository**:
   - README: "Analytics Agent" — what it does, how to deploy, example queries
   - `/examples` folder with 5-10 workflow examples (queries + outputs)
   - `ARCHITECTURE.md` explaining MCP + LangGraph design
   - `DEPLOYMENT.md` with setup, environment variables, security checklist

### Success Criteria

| Criterion | Threshold | Assessment |
|-----------|-----------|------------|
| MCP Tools | 3+ custom tools, fully spec'd | Tools have clear schemas, input validation |
| Agent Workflows | 3+ distinct workflows | Each handles different analytics task |
| Reasoning Quality | Agent uses multi-step reasoning | Traces show chain-of-thought |
| Error Handling | Graceful failure for edge cases | No crashes; clear error messages |
| Production Readiness | Logging, monitoring, cost tracking | Operational dashboard functional |
| Documentation | Architecture + deployment guide | Clear for new engineer to understand |

### Example Use Cases for Agentic System

1. **Company Existing Challenge** (preferred if available):
   - "Investigate sales decline in Q4 2024"
   - "Identify top 10 customer churn risk factors"
   - "Recommend pricing optimization"

2. **Domain-Specific Agents** (if no company challenge):
   - **Energy Analytics Agent**: Forecast demand, identify anomalies, recommend load balancing
   - **Financial Agent**: Analyze portfolio performance, identify trading opportunities
   - **E-commerce Agent**: Analyze customer behavior, recommend promotions, forecast inventory

---

## PROJECT 3: Full-Stack ETL Pipeline with Streamlit Dashboard (Weeks 10-12)

### Project Overview

Students will build an **end-to-end data platform** that ingests raw data from APIs, transforms it using concurrent ETL patterns, stores it in a warehouse, and exposes insights via a production-grade Streamlit dashboard. This mirrors real data engineering work at startups and enterprises.

### Business Context

Data engineering is the foundation of analytics and ML. Most companies struggle with data quality, freshness, and availability. This project teaches how to build reliable, scalable data systems that support downstream analytics, BI, and ML workloads.

### Learning Outcomes

- **API Integration**: Handling rate limits, pagination, retries, credential rotation
- **Concurrent ETL**: Python async/threading, parallel processing to optimize throughput
- **Data Quality**: Validation, schema enforcement, anomaly detection
- **Transformation Logic**: dbt or Python-based transformations, business logic
- **Warehouse Optimization**: Partitioning, indexing, query optimization
- **Dashboard Development**: Streamlit for interactive, performant analytics UIs
- **Operations**: Monitoring, alerting, failure recovery, documentation

### Tech Stack

| Component | Tools | Purpose |
|-----------|-------|---------|
| **Data Ingestion** | Python (requests, aiohttp for async), Prefect/Airflow | API polling, orchestration |
| **Concurrency** | asyncio, ThreadPoolExecutor | Parallel API calls, data processing |
| **Transformation** | dbt or Python (pandas, Polars) | SQL or Python-based transformations |
| **Data Warehouse** | Snowflake, BigQuery, or DuckDB | Target analytical system |
| **Dashboard** | Streamlit | Interactive analytics UI |
| **Monitoring** | Python logging, Prometheus (optional) | Pipeline health checks |

### Phase A: ETL Architecture & API Integration (Week 10)

**Mentor Checkpoint**: API design review; concurrency patterns validation; error handling

**Deliverables**:
1. **Data Source Analysis**:
   - Identify 2-3 public APIs or company data sources
   - Document: endpoints, rate limits, authentication, pagination, data schema
   - Example sources:
     - Weather API (OpenWeatherMap)
     - Stock prices (Finnhub, Alpha Vantage)
     - Public energy/utility data
     - GitHub stats, social media metrics, etc.

2. **ETL Architecture Diagram**:
   ```
   API Source 1 ─┐
   API Source 2 ─┼→ Async Fetcher → Validation → Transform → Snowflake → Streamlit
   API Source 3 ─┘                                                   ↑
                                                             Refresh on schedule
   ```

3. **Concurrent Data Fetcher** (Python code):
   ```python
   import asyncio
   from aiohttp import ClientSession
   
   async def fetch_from_apis(sources: List[str], batch_size: int = 10):
       """Fetch from multiple APIs concurrently with rate limit handling"""
       async with ClientSession() as session:
           tasks = [fetch_one(session, source) for source in sources]
           results = await asyncio.gather(*tasks, return_exceptions=True)
       return results
   
   async def fetch_one(session, source):
       """Single API call with retry + backoff"""
       for attempt in range(3):
           try:
               async with session.get(source['url'], headers=auth) as resp:
                   if resp.status == 429:  # Rate limited
                       await asyncio.sleep(2 ** attempt)
                       continue
                   return await resp.json()
           except Exception as e:
               logger.warning(f"Fetch failed attempt {attempt}: {e}")
               await asyncio.sleep(2 ** attempt)
       raise FetchError(f"Failed to fetch {source}")
   ```

4. **Validation Schema**:
   - Define expected columns, data types, null allowance
   - Use Great Expectations or Pydantic for validation
   - Log validation failures; quarantine bad records

5. **Error Handling & Retry Strategy**:
   - Rate limits (429): exponential backoff + jitter
   - Timeouts: retry up to 3 times with increasing wait
   - Invalid data: log and skip, alert on high failure rate
   - Network errors: circuit breaker pattern (fail fast after N consecutive failures)

### Phase B: Transformation & Load (Week 11)

**Mentor Checkpoint**: SQL/transformation logic review; data quality checks; warehouse setup

**Deliverables**:
1. **Data Warehouse Schema** (Snowflake/BigQuery):
   ```sql
   -- Raw layer (minimal transformation)
   CREATE TABLE raw_api_data (
       ingestion_timestamp TIMESTAMP,
       source_name STRING,
       raw_payload JSON
   );
   
   -- Transformed layer
   CREATE TABLE transformed_metrics (
       date DATE,
       metric_name STRING,
       value FLOAT,
       region STRING,
       source STRING,
       loaded_at TIMESTAMP
   );
   ```

2. **Transformation Pipeline** (dbt or Python):
   - **dbt approach**:
     ```sql
     -- models/stg_metrics.sql
     WITH raw_data AS (
         SELECT JSON_EXTRACT(raw_payload, '$.metric') AS metric,
                JSON_EXTRACT(raw_payload, '$.value') AS value,
                ingestion_timestamp
         FROM raw_api_data
     ),
     validated AS (
         SELECT *,
                CASE WHEN value IS NOT NULL THEN 1 ELSE 0 END AS is_valid
         FROM raw_data
     )
     SELECT * FROM validated
     WHERE is_valid = 1
     ```
   
   - **Python approach**:
     ```python
     def transform_data(raw_df):
         # Normalize column names
         df = raw_df.rename(columns=str.lower)
         # Type conversions
         df['value'] = pd.to_numeric(df['value'], errors='coerce')
         # Fill missing with forward fill
         df['value'] = df['value'].fillna(method='ffill')
         # Add derived columns
         df['date'] = pd.to_datetime(df['timestamp']).dt.date
         return df
     ```

3. **Data Quality Framework**:
   - Great Expectations tests:
     - Expect column to exist
     - Expect value in range [min, max]
     - Expect no null values in key columns
     - Expect values to be unique (if applicable)
   - Generate data quality report per load

4. **Incremental Loading Strategy**:
   - Upsert pattern: new records insert, existing update (based on composite key)
   - Change Data Capture (CDC): track which records changed
   - Backfill strategy: how to re-process historical data if transformation logic changes

5. **Scheduler (Airflow or Prefect)**:
   ```python
   from prefect import flow, task
   
   @task
   def fetch_task():
       return fetch_from_apis(...)
   
   @task
   def transform_task(raw_data):
       return transform_data(raw_data)
   
   @task
   def load_task(transformed_data):
       load_to_warehouse(transformed_data)
   
   @flow
   def etl_pipeline():
       raw = fetch_task()
       transformed = transform_task(raw)
       load_task(transformed)
   
   # Schedule: Run every 6 hours
   flow.serve(name="etl-pipeline", cron="0 */6 * * *")
   ```

### Phase C: Dashboard & Operations (Week 12)

**Mentor Checkpoint**: Dashboard performance; monitoring alerts; runbook review

**Deliverables**:
1. **Streamlit Dashboard** (production-ready):
   ```python
   # app/dashboard.py
   import streamlit as st
   import snowflake.connector
   import plotly.express as px
   
   st.set_page_config(page_title="Data Platform", layout="wide")
   
   @st.cache_resource
   def get_connection():
       return snowflake.connector.connect(**st.secrets["snowflake"])
   
   conn = get_connection()
   
   # Sidebar filters
   st.sidebar.header("Filters")
   date_range = st.sidebar.date_input("Date Range", value=(today - timedelta(30), today))
   region = st.sidebar.multiselect("Region", options=["North", "South", "East"])
   
   # Main content
   st.title("Analytics Dashboard")
   
   # KPIs
   col1, col2, col3 = st.columns(3)
   col1.metric("Total Records", fetch_metric("count(*)", date_range))
   col2.metric("Avg Value", fetch_metric("avg(value)", date_range))
   col3.metric("Data Freshness", fetch_metric("max(loaded_at)", date_range))
   
   # Visualizations
   st.subheader("Trends")
   df = fetch_data("SELECT date, SUM(value) FROM metrics GROUP BY 1", date_range)
   st.line_chart(df)
   
   st.subheader("Distribution by Region")
   df_region = fetch_data("SELECT region, COUNT(*) FROM metrics GROUP BY 1", date_range)
   st.bar_chart(df_region)
   
   # Data quality section
   st.subheader("Pipeline Health")
   col1, col2 = st.columns(2)
   col1.write(f"Last load: {fetch_last_load_time()}")
   col2.write(f"Data quality score: {fetch_quality_score()}%")
   ```

2. **Performance Optimization**:
   - Caching: `@st.cache_resource` for connections, `@st.cache_data` for queries
   - Query optimization: Ensure Snowflake queries use indexes, partitions
   - Incremental refresh: Only fetch changed data, not full reload
   - Connection pooling: Reuse DB connections

3. **Monitoring & Alerting**:
   - **Pipeline Monitoring**:
     ```python
     # Check if last load was <6 hours ago
     SELECT MAX(loaded_at) FROM metrics;
     IF (now - max_loaded_at) > 6 HOURS:
         ALERT("ETL pipeline stale")
     ```
   - **Data Quality Monitoring**:
     ```python
     # Flag if >10% of records are null
     SELECT COUNT(*) WHERE value IS NULL / COUNT(*) > 0.1;
     IF pct_null > 0.1:
         ALERT("Data quality issue: high null rate")
     ```
   - **Streamlit Monitoring**:
     - Track page load time, user sessions, error rates
     - Log in CloudWatch/Datadog for debugging

4. **Operational Documentation**:
   - **README**:
     - What this pipeline does
     - How to deploy locally
     - How to deploy to production
     - Dashboard walkthrough
   
   - **ARCHITECTURE.md**:
     - Data flow diagram
     - Schema documentation (table descriptions, key columns)
     - Transformation logic explanation
   
   - **RUNBOOK.md**:
     - Common issues + troubleshooting steps
     - Manual rerun procedures
     - How to backfill data if pipeline fails
     - Escalation contacts
   
   - **DEPLOYMENT.md**:
     - Environment setup (Python, dependencies)
     - Credentials/secrets management
     - Scheduling setup (Airflow/Prefect)
     - Monitoring dashboard setup

5. **GitHub Repository Structure**:
   ```
   etl-pipeline/
   ├── README.md                 # Project overview
   ├── ARCHITECTURE.md           # System design
   ├── DEPLOYMENT.md             # How to deploy
   ├── RUNBOOK.md                # Operational guide
   ├── requirements.txt          # Python dependencies
   ├── .env.example              # Environment template
   ├── src/
   │   ├── fetcher/
   │   │   └── api_client.py     # Concurrent API fetcher
   │   ├── transform/
   │   │   ├── models.py         # dbt or transform functions
   │   │   └── validation.py     # Great Expectations tests
   │   ├── load/
   │   │   └── warehouse.py      # Snowflake loading logic
   │   ├── orchestration/
   │   │   └── dag.py            # Airflow/Prefect DAG
   │   └── monitoring/
   │       └── checks.py         # Data quality checks
   ├── app/
   │   ├── dashboard.py          # Streamlit app
   │   └── pages/                # Multi-page dashboards (optional)
   ├── tests/
   │   ├── test_fetcher.py
   │   ├── test_transform.py
   │   └── test_load.py
   ├── Dockerfile
   ├── docker-compose.yml        # Local dev setup
   └── .github/
       └── workflows/
           └── ci-cd.yml         # GitHub Actions pipeline
   ```

### Success Criteria

| Criterion | Threshold | Assessment |
|-----------|-----------|------------|
| Concurrent Fetching | 3+ APIs fetched in parallel | Throughput > serial approach |
| Data Quality | >95% records pass validation | Schema and business logic checks |
| Transformation | Clear separation of raw → transformed | Documented logic, idempotent |
| Dashboard UX | <3s page load time | Interactive, responsive to filters |
| Monitoring | Alerts fire on failures/staleness | Ops team notified of issues |
| Documentation | README + Architecture + Runbook | Clear enough for new engineer |
| Code Quality | >60% test coverage | Modular, no code duplication |

---

## Summary: 3-Project Structure

| Project | Duration | Focus | Key Skills | Outcome |
|---------|----------|-------|-----------|---------|
| **Time Series** | Weeks 2-4 | Analysis + ML | Hypothesis testing, statistical rigor, Snowflake, Python ML | Deployed forecasting API |
| **Agents** | Weeks 5-9 | Reasoning + Autonomy | MCP tools, LangChain, multi-step reasoning, production LLMs | Deployed agentic system for analytics |
| **ETL Dashboard** | Weeks 10-12 | Engineering + Data Quality | Concurrent APIs, dbt/transforms, Streamlit, operations | Live production dashboard |

---

## Progression Logic

Each project builds in complexity and production maturity:

1. **Project 1 (Time Series)**: Deep analysis, rigorous statistics, introduces deployment concepts
2. **Project 2 (Agents)**: Complex reasoning, tool design, production observability
3. **Project 3 (ETL)**: End-to-end systems thinking, operational excellence, real-world data challenges

By Week 12, students have **3 portfolio pieces** that demonstrate:
- Technical depth (time series + ML + modern AI)
- Systems thinking (full-stack architecture)
- Production maturity (monitoring, error handling, documentation)

This is highly competitive for mid-level data science / ML engineering roles.

---

**Version**: 1.0 | **Last Updated**: January 2026
