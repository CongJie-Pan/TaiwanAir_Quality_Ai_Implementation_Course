# Streamlit Application Development Tasks
**The streamlit ui please use 繁體中文 to perform.**

## Phase 2: Streamlit Basic Framework

### [ ] **Task ID**: ST-001
- **Task Name**: Build Streamlit Basic Framework
- **Work Description**:
    - Why: To establish the basic structure of the Streamlit application.
    - How: Create the main program structure, implement the navigation system, complete the Sidebar control panel, and implement Session State management.
- **Resources Required**:
    - Materials: `streamlit`, `streamlit-navigation-bar`
    - Personnel: Developer
    - Reference Codes/docs: `docs/PLANNING.md` (Section 4), `CourseCode/AIp04空間與網站X.py`
- **Deliverables**:
    - [ ] Main application structure created.
    - [ ] Navigation system implemented.
    - [ ] Sidebar control panel completed.
    - [ ] Session State management implemented.
- **Dependencies**: Phase 1 (Data backend) completed.
- **Constraints**: Must follow the design in `PLANNING.md`.
- **Completion Status**: Not Started
- **Notes**: This is the foundation for all other UI work.

## Phase 3: Data & Information Layer

### [ ] **Task ID**: ST-002
- **Task Name**: Implement Data and Information Layer Pages
- **Work Description**:
    - Why: To display the raw data and basic statistical information.
    - How: Create "Data Overview" and "Statistical Analysis" pages. Implement basic visualization and crosstab analysis.
- **Resources Required**:
    - Materials: `pandas`, `plotly`
    - Personnel: Developer
    - Reference Codes/docs: `docs/PLANNING.md` (Section 3.2, 3.3, 4.3)
- **Deliverables**:
    - [ ] "Data Overview" page completed.
    - [ ] "Statistical Analysis" page completed.
    - [ ] Basic visualization functions implemented.
    - [ ] Crosstab analysis feature implemented.
- **Dependencies**: ST-001
- **Constraints**: Visualizations should be clear and interactive.
- **Completion Status**: Not Started
- **Notes**: Focus on providing a clear overview of the data.

## Phase 4: Knowledge & Wisdom Layer

### [ ] **Task ID**: ST-003
- **Task Name**: Implement Knowledge and Wisdom Layer Pages
- **Work Description**:
    - Why: To uncover patterns in the data and provide actionable insights.
    - How: Create "Pattern Discovery" and "Wisdom Decision" pages. Implement correlation analysis and a health recommendation system.
- **Resources Required**:
    - Materials: `pandas`, `plotly`
    - Personnel: Developer
    - Reference Codes/docs: `docs/PLANNING.md` (Section 3.4, 3.5, 4.3)
- **Deliverables**:
    - [ ] "Pattern Discovery" page completed.
    - [ ] "Wisdom Decision" page completed.
    - [ ] Correlation analysis feature implemented.
    - [ ] Health recommendation system implemented.
- **Dependencies**: ST-002
- **Constraints**: Insights should be easy to understand for non-technical users.
- **Completion Status**: Not Started
- **Notes**: This is where the DIKW model provides the most value.

## Phase 5: Prediction Model

### [ ] **Task ID**: ST-004
- **Task Name**: Implement Prediction Model Page (Only Perform the front-end 架構, not do the acutal back-end(model). )
- **Work Description**:
    - Why: To provide future forecasts of air quality.
    - How: Create a "Prediction Model" page. Implement time series forecasting and scenario simulation.
- **Resources Required**:
    - Materials: `scikit-learn`, `prophet` (optional)
    - Personnel: Data Scientist/Developer
    - Reference Codes/docs: `docs/PLANNING.md` (Section 5.4, 4.3)
- **Deliverables**:
    - [ ] "Prediction Model" page completed.
    - [ ] Time series forecasting implemented.
    - [ ] Scenario simulation feature implemented.
    - [ ] Model performance evaluation displayed.
- **Dependencies**: ST-003
- **Constraints**: Predictions should be presented with uncertainty intervals.
- **Completion Status**: Not Started
- **Notes**: Start with a simple model and iterate.

## Phase 6: Optimization & Deployment

### [ ] **Task ID**: ST-005
- **Task Name**: Optimize and Deploy the Streamlit Application(Not do this one first.)
- **Work Description**:
    - Why: To improve the performance and user experience, and make the application accessible.
    - How: Optimize performance, improve UI/UX, complete documentation, and deploy to the cloud.
- **Resources Required**:
    - Materials: `Streamlit Cloud`, `Docker` (optional)
    - Personnel: DevOps/Developer
    - Reference Codes/docs: `docs/PLANNING.md` (Section 6.1)
- **Deliverables**:
    - [ ] Performance optimization completed.
    - [ ] UI/UX improvements implemented.
    - [ ] Documentation finalized.
    - [ ] Application deployed to Streamlit Cloud.
- **Dependencies**: ST-004
- **Constraints**: The deployed application should be stable and responsive.
- **Completion Status**: Not Started
- **Notes**: Final step to make the project live.

---

## Discovered During Work

- [2025-10-14] Fix data load error: duplicate labels in time period binning caused pandas error "labels must be unique if ordered=True". Resolved by using unique labels in `utils/app_utils.py` (`凌晨`, `早晨`, `上午`, `下午`, `傍晚`, `夜間`). [Completed]
- [2025-10-14] Remove sidebar user input feature (username/suggestion) per requirement. Cleaned session state and removed UI block in `app.py`. [Completed]
- [2025-10-14] Expand usage instructions detailing how to operate the control panel in `app.py` (empty-state help) and `README.md` (Streamlit App section). [Completed]
- [2025-10-14] Temporarily hide pages 4 and 5 (智慧決策、預測模型): removed from navigation via `HIDE_ADVANCED_PAGES=True` in `src/main/python/app.py` and stopped exporting them in `src/main/python/pages/__init__.py`. Restore by setting flag to False and re-exporting. [Completed]
- [2025-10-14] Set default sidebar date range to 2023/01/01–2023/12/31, clamped to available data; fallback to full available range when 2023 is out of bounds. Implemented in `src/main/python/app.py`. [Completed]
