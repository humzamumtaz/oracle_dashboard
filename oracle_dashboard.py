
import pandas as pd
import pytz
import urllib.parse
from datetime import datetime, timedelta, time
from sqlalchemy import create_engine, text, pool
from sqlalchemy.exc import SQLAlchemyError
import streamlit as st
import traceback
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import threading
import io
import concurrent.futures
import logging
import time as time_module

# --- Page Configuration (Best practice: call this first) ---
st.set_page_config(
    page_title="Manufacturing KPI Dashboard",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Session State Initialization ---
def initialize_session_state():
    """Initialize all session state variables for persistent filters and UI state."""
    defaults = {
        'selected_page': "📊 KPI Dashboard",
        'last_refresh': datetime.now(),
        'cached_data': None,
        'data_load_error': False,
        'maximized': False,
        'filter_program': [],
        'filter_line': [],
        'filter_user': [],
        'filter_operation': [],
        'filter_preset': 'None',
        'filter_min_uph': 0.0,
        'filter_show_inactive': True,
        'filter_date_range': None,
        'auto_refresh_enabled': False,
        'page_load_times': {},
        'error_count': 0,
        'kpi_history': [],
        'hourly_data': pd.DataFrame()
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# --- Enhanced Database Connection with Better Error Handling ---
@st.cache_resource
def create_db_engine():
    """Create database engine with improved connection pooling and error handling."""
    username = st.secrets["database"]["username"]
    raw_password = st.secrets["database"]["password"]
    password = urllib.parse.quote(raw_password, safe="")  # This line encodes special characters
    host = st.secrets["database"]["host"]
    port = st.secrets["database"]["port"]
    service = st.secrets["database"]["service"]

    connection_string = f"oracle+cx_oracle://{username}:{password}@{host}:{port}/?service_name={service}"
    
    try:
        engine = create_engine(
            connection_string,
            poolclass=pool.QueuePool,
            pool_size=3,  # Reduced for stability
            max_overflow=5,
            pool_pre_ping=True,
            pool_recycle=1800,  # 30 minutes
            pool_timeout=30,
            echo=False
        )
        # Test connection
        with engine.connect() as conn:
            conn.execute(text("SELECT 1 FROM DUAL"))
        return engine
    except Exception as e:
        st.error(f"Database connection failed: {e}")
        logger.error(f"Database connection error: {e}")
        return None

engine = create_db_engine()

# --- Time and Shift Functions (Consolidated and Robust) ---
def get_central_time_now():
    """Get current Central Time with accurate DST handling using pytz."""
    try:
        central_tz = pytz.timezone("America/Chicago")
        return datetime.now(central_tz)
    except Exception:
        # Fallback to manual UTC offset calculation if pytz fails
        utc_now = datetime.utcnow()
        if (datetime(utc_now.year, 3, 14) <= utc_now.replace(hour=0, minute=0, second=0, microsecond=0) < datetime(utc_now.year, 11, 7)):
            return utc_now - timedelta(hours=5)
        return utc_now - timedelta(hours=6)

def get_current_shift():
    """Determine current shift based on Central Time."""
    central_now = get_central_time_now()
    current_time = central_now.time()
    current_day_of_week = central_now.weekday()  # Monday is 0, Sunday is 6

    # Monday to Thursday, 5:45 AM - 5:00 PM
    if 0 <= current_day_of_week <= 3 and time(5, 45) <= current_time < time(17, 0):
        return "1st Shift"
    
    # Monday to Thursday night shifts, wrapping around midnight
    if (0 <= current_day_of_week <= 2 and current_time >= time(17, 0)) or \
       (1 <= current_day_of_week <= 3 and current_time < time(4, 0)):
        return "2nd Shift"
    
    # Weekend Shift, Friday to Sunday, 5:45 AM - 6:30 PM
    if 4 <= current_day_of_week <= 6 and time(5, 45) <= current_time < time(18, 30):
        return "3rd Shift"

    return "Off Shift"

def get_shift_window():
    """Get shift start and end times for the CURRENT shift, returns AWARE datetime objects."""
    central_now = get_central_time_now()
    current_shift = get_current_shift()
    
    if current_shift == "1st Shift":
        start = central_now.replace(hour=5, minute=45, second=0, microsecond=0)
        end = central_now.replace(hour=17, minute=0, second=0, microsecond=0)
    elif current_shift == "2nd Shift":
        if central_now.time() >= time(17, 30):
            start = central_now.replace(hour=17, minute=30, second=0, microsecond=0)
            end = (central_now + timedelta(days=1)).replace(hour=4, minute=0, second=0, microsecond=0)
        else:
            start = (central_now - timedelta(days=1)).replace(hour=17, minute=30, second=0, microsecond=0)
            end = central_now.replace(hour=4, minute=0, second=0, microsecond=0)
    elif current_shift == "3rd Shift":
        start = central_now.replace(hour=5, minute=45, second=0, microsecond=0)
        end = central_now.replace(hour=18, minute=30, second=0, microsecond=0)
    else:  # Off Shift, get a recent window
        start = central_now - timedelta(hours=12)
        end = central_now
    
    return start, end

# --- Queries from Working Code.txt ---
QUERIES = {
    "item_receipt": """
        WITH MaxFlowOps AS (
            SELECT x1.*
            FROM xxaiz.xxaiz_rec_operation_dtl_tbl x1
            WHERE x1.OPERATION_FLOW_SEQ = (
                SELECT MAX(x2.OPERATION_FLOW_SEQ)
                FROM xxaiz.xxaiz_rec_operation_dtl_tbl x2
                WHERE x2.WIP_ENTITY_ID = x1.WIP_ENTITY_ID
            )
        )
        SELECT DISTINCT 
            IRHT.ITEM_DESC,
            wdj.WIP_ENTITY_NAME,
            wdj.WIP_ENTITY_ID,
            wdj.ATTRIBUTE8 AS Program,
            wdj.ATTRIBUTE2 AS IMEI,
            wdj.ATTRIBUTE1 AS INSTANCE_NUMBER,
            wo.DESCRIPTION AS WIP_JOB,
            wdj.PRIMARY_ITEM_ID,
            wdj.STATUS_TYPE,
            wdj.ATTRIBUTE6,
            wo.OPERATION_SEQ_NUM,
            wdj.CREATION_DATE,
            wo.DATE_LAST_MOVED AS completion_date,
            xrodt.OPERATION_FLOW_SEQ,
            xrodt.WORKSTATION,
            xrodt.OPERATION_COMPLETED_BY,
            usr.DESCRIPTION AS USER_DESCRIPTION
        FROM apps.wip_operations wo 
        JOIN apps.wip_discrete_jobs_v wdj 
            ON wo.WIP_ENTITY_ID = wdj.WIP_ENTITY_ID
        LEFT JOIN MaxFlowOps xrodt 
            ON wo.WIP_ENTITY_ID = xrodt.WIP_ENTITY_ID
        LEFT JOIN apps.fnd_user usr 
            ON xrodt.OPERATION_COMPLETED_BY = usr.USER_ID
        LEFT JOIN xxaiz.XXAIZ_ITEM_RECPT_HDR_TBL IRHT 
            ON IRHT.INSTANCE_NO = wdj.ATTRIBUTE1
        WHERE wo.date_last_moved BETWEEN :start_time AND :end_time
          AND wo.DESCRIPTION = 'Lewisville RCV Item Receipt'
    """,
    
    "in_queue": """
        WITH MaxFlowOps AS (
            SELECT x1.*
            FROM xxaiz.xxaiz_rec_operation_dtl_tbl x1
            WHERE x1.OPERATION_FLOW_SEQ = (
                SELECT MAX(x2.OPERATION_FLOW_SEQ)
                FROM xxaiz.xxaiz_rec_operation_dtl_tbl x2
                WHERE x2.WIP_ENTITY_ID = x1.WIP_ENTITY_ID
            )
        )
        SELECT DISTINCT 
            IRHT.ITEM_DESC,
            wdj.WIP_ENTITY_NAME,
            wdj.WIP_ENTITY_ID,
            wdj.ATTRIBUTE8 AS Program,
            wdj.ATTRIBUTE2 AS IMEI,
            wdj.ATTRIBUTE1 AS INSTANCE_NUMBER,
            wo.DESCRIPTION AS WIP_JOB,
            wdj.PRIMARY_ITEM_ID,
            wdj.STATUS_TYPE,
            wdj.ATTRIBUTE6,
            wo.OPERATION_SEQ_NUM,
            wdj.CREATION_DATE,
            wo.DATE_LAST_MOVED AS completion_date,
            xrodt.OPERATION_FLOW_SEQ,
            xrodt.WORKSTATION,
            xrodt.OPERATION_COMPLETED_BY,
            usr.DESCRIPTION AS USER_DESCRIPTION
        FROM apps.wip_operations wo 
        JOIN apps.wip_discrete_jobs_v wdj 
            ON wo.WIP_ENTITY_ID = wdj.WIP_ENTITY_ID
        LEFT JOIN MaxFlowOps xrodt 
            ON wo.WIP_ENTITY_ID = xrodt.WIP_ENTITY_ID
        LEFT JOIN apps.fnd_user usr 
            ON xrodt.OPERATION_COMPLETED_BY = usr.USER_ID
        LEFT JOIN xxaiz.XXAIZ_ITEM_RECPT_HDR_TBL IRHT 
            ON IRHT.INSTANCE_NO = wdj.ATTRIBUTE1
        WHERE wo.FIRST_UNIT_START_DATE BETWEEN :start_time AND :end_time
          AND wo.DESCRIPTION LIKE 'Lewisville RCV%'
          AND wo.QUANTITY_IN_QUEUE = 1
    """,
    
    "receipt_complete": """
        WITH MaxFlowOps AS (
            SELECT x1.*
            FROM xxaiz.xxaiz_rec_operation_dtl_tbl x1
            WHERE x1.OPERATION_FLOW_SEQ = (
                SELECT MAX(x2.OPERATION_FLOW_SEQ)
                FROM xxaiz.xxaiz_rec_operation_dtl_tbl x2
                WHERE x2.WIP_ENTITY_ID = x1.WIP_ENTITY_ID
            )
        )
        SELECT DISTINCT 
            IRHT.ITEM_DESC,
            wdj.WIP_ENTITY_NAME,
            wdj.WIP_ENTITY_ID,
            IRHT.PROGRAM_CODE AS Program,
            wdj.ATTRIBUTE2 AS IMEI,
            wdj.ATTRIBUTE1 AS INSTANCE_NUMBER,
            wo.DESCRIPTION AS WIP_JOB,
            wdj.PRIMARY_ITEM_ID,
            wdj.STATUS_TYPE,
            wdj.ATTRIBUTE6,
            wo.OPERATION_SEQ_NUM,
            wdj.CREATION_DATE,
            wo.DATE_LAST_MOVED AS completion_date,
            xrodt.OPERATION_FLOW_SEQ,
            xrodt.WORKSTATION,
            xrodt.OPERATION_COMPLETED_BY,
            usr.DESCRIPTION AS USER_DESCRIPTION
        FROM apps.wip_operations wo 
        JOIN apps.wip_discrete_jobs_v wdj 
            ON wo.WIP_ENTITY_ID = wdj.WIP_ENTITY_ID
        LEFT JOIN MaxFlowOps xrodt 
            ON wo.WIP_ENTITY_ID = xrodt.WIP_ENTITY_ID
        LEFT JOIN xxaiz.xxaiz_item_RECPT_HDR_TBL IRHT 
            ON IRHT.INSTANCE_NO = wdj.ATTRIBUTE1
        LEFT JOIN apps.fnd_user usr 
            ON xrodt.OPERATION_COMPLETED_BY = usr.USER_ID
        WHERE wo.date_last_moved BETWEEN :start_time AND :end_time
          AND wo.DESCRIPTION = 'Lewisville RCV Receipt Complete'
    """,
    
    "unknown_analysis": """
        SELECT DISTINCT 
            IRHT.ITEM_DESC,
            wdj.WIP_ENTITY_NAME,
            wdj.WIP_ENTITY_ID,
            wdj.ATTRIBUTE8 AS Program,
            wdj.ATTRIBUTE2 AS IMEI,
            wdj.ATTRIBUTE1 AS INSTANCE_NUMBER,
            wo.DESCRIPTION AS WIP_JOB,
            wdj.PRIMARY_ITEM_ID,
            wdj.STATUS_TYPE,
            wdj.ATTRIBUTE6,
            wo.OPERATION_SEQ_NUM,
            wdj.CREATION_DATE,
            wo.DATE_LAST_MOVED AS completion_date,
            xrodt.OPERATION_FLOW_SEQ,
            xrodt.WORKSTATION,
            xrodt.OPERATION_COMPLETED_BY,
            usr.DESCRIPTION AS USER_DESCRIPTION,
            recovery.STATUS as RECOVERY_STATUS,
            recovery.IMEI_NO as RECOVERY_IMEI
        FROM apps.wip_operations wo 
        JOIN apps.wip_discrete_jobs_v wdj 
            ON wo.WIP_ENTITY_ID = wdj.WIP_ENTITY_ID
        LEFT JOIN xxaiz.xxaiz_rec_operation_dtl_tbl xrodt 
            ON wo.WIP_ENTITY_ID = xrodt.WIP_ENTITY_ID
        LEFT JOIN apps.fnd_user usr 
            ON xrodt.OPERATION_COMPLETED_BY = usr.USER_ID
        LEFT JOIN xxaiz.XXAIZ_ITEM_RECPT_HDR_TBL IRHT 
            ON IRHT.INSTANCE_NO = wdj.ATTRIBUTE1
        LEFT JOIN XXAIZ.XXAIZ_ITEM_RECPT_HDR_TBL recovery
            ON recovery.IMEI_NO = wdj.ATTRIBUTE2
            AND recovery.STATUS IN ('N','MC')
        WHERE wo.FIRST_UNIT_START_DATE BETWEEN :start_time AND :end_time
          AND wo.DESCRIPTION = 'Lewisville RCV Unknown Hold'
          AND wo.QUANTITY_IN_QUEUE = 1
    """,
    
    "programs": """
        SELECT
            FLV.LOOKUP_CODE AS Program,
            FLV.MEANING AS Meaning,
            FLV.DESCRIPTION AS Description
        FROM APPS.FND_LOOKUP_VALUES_VL FLV
        WHERE 1=1
          AND FLV.LOOKUP_TYPE = 'XXAIZ_PROGRAM_CP_MAP_LKP'
        ORDER BY FLV.LOOKUP_CODE
    """
}

# --- Enhanced Data Loading with Robust Fallback ---
def load_data_with_robust_fallback():
    """Enhanced data loading with better error handling and retry logic"""
    max_retries = 3
    retry_delay = 2
    
    progress_placeholder = st.empty()
    
    for attempt in range(max_retries):
        try:
            progress_placeholder.info(f"⏳ Loading data... (Attempt {attempt + 1}/{max_retries})")
            
            start_time, end_time = get_shift_window()
            dfs = {}
            
            # Load data sequentially with individual error handling
            for name, query in QUERIES.items():
                try:
                    params = {"start_time": start_time, "end_time": end_time} if ":start_time" in query else {}
                    dfs[name] = pd.read_sql(text(query), engine, params=params)
                    progress_placeholder.info(f"✅ Loaded {name}: {len(dfs[name])} records")
                except Exception as query_error:
                    st.warning(f"⚠️ Failed to load {name}: {query_error}")
                    dfs[name] = pd.DataFrame()
            
            progress_placeholder.empty()
            st.session_state.cached_data = (dfs, start_time, end_time)
            st.session_state.data_load_error = False
            st.session_state.last_refresh = datetime.now()
            return dfs, start_time, end_time
            
        except Exception as e:
            if attempt < max_retries - 1:
                progress_placeholder.warning(f"⚠️ Attempt {attempt + 1} failed: {e}. Retrying in {retry_delay}s...")
                time_module.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                progress_placeholder.empty()
                st.session_state.data_load_error = True
                if st.session_state.cached_data is not None:
                    st.warning(f"⚠️ All attempts failed. Using cached data from {st.session_state.last_refresh.strftime('%H:%M:%S')}")
                    return st.session_state.cached_data
                st.error(f"❌ Database connection failed after {max_retries} attempts: {e}")
                return {}, None, None

@st.cache_data(show_spinner=False, ttl=900, max_entries=3)
def load_data():
    """Cached wrapper for data loading to prevent unnecessary re-execution."""
    return load_data_with_robust_fallback()

# --- Optimized Data Processing Functions ---
def optimized_data_processing(dfs):
    """Optimized data processing with vectorized operations and proper concat handling"""
    if not dfs:
        return pd.DataFrame()
    
    # Pre-allocate list for better performance
    dataframes_to_concat = []
    
    for name in ["in_queue", "item_receipt", "receipt_complete"]:
        if name in dfs and not dfs[name].empty:
            df = dfs[name].copy()
            df["source"] = name
            # Vectorized column cleaning
            df.columns = df.columns.str.strip().str.lower()
            dataframes_to_concat.append(df)
    
    if not dataframes_to_concat:
        return pd.DataFrame()
    
    # FIXED: Handle the concat deprecation warning
    # Use ignore_index=True and explicitly handle empty columns
    try:
        df_all = pd.concat(dataframes_to_concat, ignore_index=True, sort=False)
        
        # Optional: Remove completely empty columns if desired (future-proof)
        # df_all = df_all.dropna(axis=1, how='all')
        
    except FutureWarning:
        # For future pandas versions, be explicit about empty column handling
        df_all = pd.concat(dataframes_to_concat, ignore_index=True, sort=False)
        # Explicitly drop all-NA columns if that's the desired behavior
        df_all = df_all.dropna(axis=1, how='all')
    
    # Optimize program mapping with vectorized operations
    if "programs" in dfs and not dfs["programs"].empty:
        program_map = dfs["programs"].copy()
        program_map.columns = program_map.columns.str.strip().str.lower()
        
        # Vectorized string operations
        program_map["program"] = program_map["program"].astype(str).str.strip().str.upper()
        df_all["program"] = df_all["program"].astype(str).str.strip().str.upper()
        
        # Use merge instead of multiple operations
        df_all = df_all.merge(
            program_map[["program", "meaning"]].drop_duplicates(), 
            on="program", 
            how="left"
        )
        df_all["program_description"] = df_all["meaning"].fillna(df_all["program"])
        df_all.drop("meaning", axis=1, inplace=True)
    else:
        df_all["program_description"] = df_all.get("program", pd.Series(dtype='str'))
    
    return df_all

def rank_latest_imei(df_all):
    """Get the latest record for each IMEI based on operation sequence number."""
    if df_all.empty or "imei" not in df_all.columns:
        return df_all
    
    # Sort and deduplicate - no concat operations here, should be fine
    df_all = df_all.sort_values("operation_seq_num", ascending=False)
    df_master = df_all.drop_duplicates(subset="imei", keep="first")
    return df_master

def add_unknown_rework_column(df_master, df_unknown):
    """Add rework column from unknown analysis based on IMEI matching."""
    if df_unknown.empty:
        df_master["unknown_rework"] = None
        return df_master
    
    # Clean column names consistently
    df_unknown.columns = df_unknown.columns.str.strip().str.lower()
    df_master.columns = df_master.columns.str.strip().str.lower()
    
    # Process unknown data
    df_latest_unknown = (
        df_unknown
        .sort_values("operation_seq_num", ascending=False)
        .drop_duplicates(subset="imei", keep="first")[["imei", "recovery_status"]]
    )
    
    # Merge operation - this should be fine
    df_master = df_master.merge(df_latest_unknown, on="imei", how="left")
    df_master.rename(columns={"recovery_status": "unknown_rework"}, inplace=True)
    return df_master

def fill_missing_user_descriptions(df):
    """Fill missing user descriptions based on workstation activity."""
    if df.empty: 
        return df
    df.columns = df.columns.str.strip().str.lower()
    if 'user_description' not in df.columns: 
        df['user_description'] = 'Unknown User'
    missing_mask = df["user_description"].isna() | (df["user_description"].str.strip() == "")
    if 'workstation' in df.columns:
        fallback_map = (
            df[~missing_mask]
            .groupby("workstation")["user_description"]
            .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else "Unknown User")
            .to_dict()
        )
        df.loc[missing_mask, "user_description"] = df.loc[missing_mask, "workstation"].map(fallback_map)
    df["user_description_filled"] = df["user_description"].fillna("Unknown User")
    return df

# --- Target Definitions ---
TARGETS = {
    'line_uph': 120,
    'completion_rate': 90,
    'unknown_rate_max': 10,
    'overall_uph': 150
}

# --- Real-Time Indicator Component ---
def create_live_status_indicator():
    """Create a live status indicator with pulsing animation"""
    return """
    <style>
    .live-indicator {
        animation: pulse 2s infinite;
        color: #28a745;
        font-weight: bold;
        display: inline-flex;
        align-items: center;
        gap: 5px;
    }
    .live-dot {
        width: 8px;
        height: 8px;
        background-color: #28a745;
        border-radius: 50%;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.5; transform: scale(0.8); }
        100% { opacity: 1; transform: scale(1); }
    }
    .connection-status {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 4px 8px;
        background: rgba(40, 167, 69, 0.1);
        border-radius: 15px;
        border: 1px solid #28a745;
        font-size: 12px;
    }
    </style>
    <div class="connection-status">
        <div class="live-dot"></div>
        <span class="live-indicator">LIVE</span>
    </div>
    """

# --- Enhanced KPI Calculations with Historical Tracking ---
def calculate_enhanced_kpis(dfs, df_master):
    """Calculate comprehensive KPIs with targets and historical tracking"""
    central_time = get_central_time_now()
    shift_start_time, _ = get_shift_window()
    hours_elapsed = max((central_time.replace(tzinfo=None) - shift_start_time.replace(tzinfo=None)).total_seconds() / 3600, 0.1)
    
    kpis = {
        'item_receipt_count': 0,
        'in_queue_count': 0,
        'receipt_complete_count': 0,
        'unknown_count': 0,
        'completion_rate': 0,
        'unknown_rate': 0,
        'throughput_rate': 0,
        'line1_devices': 0,
        'line2_devices': 0,
        'line1_uph': 0,
        'line2_uph': 0,
        'active_operators': 0,
        'timestamp': central_time
    }

    # Safe data access with validation
    for key, df_name in [('item_receipt_count', 'item_receipt'), 
                        ('in_queue_count', 'in_queue'),
                        ('receipt_complete_count', 'receipt_complete'),
                        ('unknown_count', 'unknown_analysis')]:
        if df_name in dfs and isinstance(dfs[df_name], pd.DataFrame):
            kpis[key] = len(dfs[df_name])

    # Performance metrics
    total_processed = kpis['receipt_complete_count'] + kpis['unknown_count']
    if total_processed > 0:
        kpis['completion_rate'] = (kpis['receipt_complete_count'] / total_processed) * 100
        kpis['unknown_rate'] = (kpis['unknown_count'] / total_processed) * 100
    
    kpis['throughput_rate'] = kpis['receipt_complete_count'] / hours_elapsed if hours_elapsed > 0 else 0

    # Line performance from the unified master table
    if not df_master.empty and 'workstation' in df_master.columns:
        line1_stations = [f"L01OPS.IR2.{i}" for i in range(1, 27)]
        line2_stations = [f"L01OPS.IR2.{i}" for i in range(27, 63)]
        line1_data = df_master[df_master['workstation'].isin(line1_stations)]
        line2_data = df_master[df_master['workstation'].isin(line2_stations)]
        kpis['line1_devices'] = len(line1_data)
        kpis['line2_devices'] = len(line2_data)
        kpis['line1_uph'] = kpis['line1_devices'] / hours_elapsed if hours_elapsed > 0 else 0
        kpis['line2_uph'] = kpis['line2_devices'] / hours_elapsed if hours_elapsed > 0 else 0
        kpis['line1_stations'] = line1_data['workstation'].nunique() if not line1_data.empty else 0
        kpis['line2_stations'] = line2_data['workstation'].nunique() if not line2_data.empty else 0
        
        if 'user_description_filled' in df_master.columns:
            kpis['active_operators'] = df_master['user_description_filled'].nunique()
    
    # Target comparisons
    kpis['completion_vs_target'] = kpis['completion_rate'] - TARGETS['completion_rate']
    kpis['unknown_vs_target'] = TARGETS['unknown_rate_max'] - kpis['unknown_rate']
    kpis['line1_vs_target'] = kpis['line1_uph'] - TARGETS['line_uph']
    kpis['line2_vs_target'] = kpis['line2_uph'] - TARGETS['line_uph']
    kpis['overall_vs_target'] = kpis['throughput_rate'] - TARGETS['overall_uph']
    
    # Store historical data for trends
    if 'kpi_history' not in st.session_state:
        st.session_state.kpi_history = []
    
    # Keep last 24 hours of data (assuming 5-minute intervals = 288 points)
    st.session_state.kpi_history.append(kpis)
    if len(st.session_state.kpi_history) > 288:
        st.session_state.kpi_history = st.session_state.kpi_history[-288:]
    
    return kpis

# --- Trend Sparkline Creation ---
def create_trend_sparkline(data, metric, color="#4472C4", height=60):
    """Create mini trend charts for KPI cards"""
    if not data or len(data) < 2:
        return go.Figure()
    
    # Extract timestamps and values
    timestamps = [item['timestamp'] for item in data[-20:]]  # Last 20 points
    values = [item.get(metric, 0) for item in data[-20:]]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=values,
        mode='lines',
        line=dict(width=2, color=color),
        hovertemplate='%{y:.1f}<extra></extra>'
    ))
    
    fig.update_layout(
        height=height,
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

# --- Enhanced KPI Card Component ---
def create_enhanced_kpi_card(title, current_value, target_value, delta, trend_data, metric_key, unit="", trend_color="#4472C4"):
    """Create enhanced KPI card with sparkline trend using proper Streamlit components"""
    
    # Determine status and colors
    if "rate" in metric_key.lower() and "unknown" in metric_key.lower():
        # For unknown rate, lower is better
        delta_color = "normal" if current_value <= target_value else "inverse"
        status_icon = "✅" if current_value <= target_value else "❌"
    else:
        # For other metrics, higher is generally better
        delta_color = "normal" if current_value >= target_value else "inverse"
        status_icon = "📈" if current_value >= target_value else "📉"
    
    # Create the metric with delta
    st.metric(
        label=f"{status_icon} {title}",
        value=f"{current_value:.1f}{unit}",
        delta=f"Target: {target_value}{unit} (Δ {delta:+.1f}{unit})",
        delta_color=delta_color
    )
    
    # Add mini trend chart if we have historical data
    if len(trend_data) > 2:
        trend_fig = create_trend_sparkline(trend_data, metric_key, trend_color, height=40)
        st.plotly_chart(trend_fig, use_container_width=True, config={'displayModeBar': False})

# --- Alert System ---
def generate_alerts(kpis):
    """Generate smart alerts based on KPIs with clear labels and colors."""
    alerts = []
    if kpis['unknown_rate'] > 20: 
        alerts.append(("CRITICAL", f"Unknown rate is extremely high at {kpis['unknown_rate']:.1f}%."))
    if kpis['completion_rate'] < 50: 
        alerts.append(("CRITICAL", f"Completion rate is critically low at {kpis['completion_rate']:.1f}%."))
    if kpis['unknown_rate'] > TARGETS['unknown_rate_max']: 
        alerts.append(("WARNING", f"Unknown rate is above target: {kpis['unknown_rate']:.1f}% (Target: {TARGETS['unknown_rate_max']}%)"))
    if kpis['completion_rate'] < TARGETS['completion_rate']: 
        alerts.append(("WARNING", f"Completion rate is below target: {kpis['completion_rate']:.1f}% (Target: {TARGETS['completion_rate']}%)"))
    if kpis.get('line1_uph', 0) < TARGETS['line_uph']: 
        alerts.append(("WARNING", f"Line 1 UPH is below target: {kpis.get('line1_uph', 0):.1f} (Target: {TARGETS['line_uph']})"))
    if kpis.get('line2_uph', 0) < TARGETS['line_uph']: 
        alerts.append(("WARNING", f"Line 2 UPH is below target: {kpis.get('line2_uph', 0):.1f} (Target: {TARGETS['line_uph']})"))
    if not alerts: 
        alerts.append(("GOOD", "All primary KPIs are within target range."))
    return alerts

# --- Enhanced KPI Overview Page ---
@st.fragment
def create_enhanced_kpi_overview_page(dfs, df_master):
    """Creates a comprehensive KPI overview page with real-time indicators and trends"""
    current_shift = get_current_shift()
    kpis = calculate_enhanced_kpis(dfs, df_master)
    alerts = generate_alerts(kpis)

    # Header with live status indicator
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #4472C4 0%, #2E5984 100%); color: white;
                    text-align: left; padding: 20px; border-radius: 8px; margin-bottom: 20px;'>
            <h2 style='margin: 0; font-size: 28px;'>🏭 Manufacturing KPI Dashboard</h2>
            <p style='margin: 5px 0 0 0; font-size: 16px; opacity: 0.9;'>{current_shift} | Live Performance Metrics</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(create_live_status_indicator(), unsafe_allow_html=True)
        st.caption(f"Last updated: {datetime.now().strftime('%H:%M:%S')}")
    
    # Alert banner
    if alerts:
        alert_colors = {"CRITICAL": "#dc3545", "WARNING": "#fd7e14", "GOOD": "#28a745"}
        for alert_type, message in alerts[:3]:
            color = alert_colors.get(alert_type, "#6c757d")
            st.markdown(f"""
            <div style='background: {color}; color: white; padding: 8px 15px; border-radius: 4px;
                        margin-bottom: 5px; font-size: 14px; font-weight: bold;'>
                {alert_type}: {message}
            </div>
            """, unsafe_allow_html=True)
        st.divider()
    
    # Enhanced KPI Cards with Trends
    st.subheader("🎯 Primary KPIs - Real-Time Performance")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        create_enhanced_kpi_card(
            "Overall Throughput", 
            kpis['throughput_rate'], 
            TARGETS['overall_uph'], 
            kpis['overall_vs_target'], 
            st.session_state.kpi_history, 
            'throughput_rate', 
            " UPH"
        )
    
    with col2:
        create_enhanced_kpi_card(
            "Completion Rate", 
            kpis['completion_rate'], 
            TARGETS['completion_rate'], 
            kpis['completion_vs_target'], 
            st.session_state.kpi_history, 
            'completion_rate', 
            "%"
        )
    
    with col3:
        create_enhanced_kpi_card(
            "Unknown Rate", 
            kpis['unknown_rate'], 
            TARGETS['unknown_rate_max'], 
            kpis['unknown_vs_target'], 
            st.session_state.kpi_history, 
            'unknown_rate', 
            "%",
            "#ff6b6b"
        )
    
    with col4:
        create_enhanced_kpi_card(
            "Line 1 UPH", 
            kpis.get('line1_uph', 0), 
            TARGETS['line_uph'], 
            kpis['line1_vs_target'], 
            st.session_state.kpi_history, 
            'line1_uph', 
            ""
        )
    
    with col5:
        create_enhanced_kpi_card(
            "Line 2 UPH", 
            kpis.get('line2_uph', 0), 
            TARGETS['line_uph'], 
            kpis['line2_vs_target'], 
            st.session_state.kpi_history, 
            'line2_uph', 
            ""
        )
    
    # Process Flow with Real-Time Counts using Streamlit columns and metrics
    st.divider()
    st.subheader("📈 Process Flow Status")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="📥 Item Receipt",
            value=f"{kpis['item_receipt_count']:,}",
            help="Devices entering the system"
        )
    
    with col2:
        st.metric(
            label="⏳ In Queue", 
            value=f"{kpis['in_queue_count']:,}",
            help="Devices currently being processed"
        )
    
    with col3:
        st.metric(
            label="✅ Receipt Complete",
            value=f"{kpis['receipt_complete_count']:,}",
            help="Successfully completed devices"
        )
    
    with col4:
        st.metric(
            label="❓ Unknown Hold",
            value=f"{kpis['unknown_count']:,}",
            delta=f"-{kpis['unknown_rate']:.1f}%" if kpis['unknown_rate'] < 10 else f"+{kpis['unknown_rate']:.1f}%",
            delta_color="normal" if kpis['unknown_rate'] < 10 else "inverse",
            help="Devices requiring attention"
        )
    
    # Hourly Trend Chart
    st.divider()
    st.subheader("📊 Hourly Throughput Trends")
    
    if len(st.session_state.kpi_history) > 5:
        # Create hourly trend chart
        trend_df = pd.DataFrame(st.session_state.kpi_history[-48:])  # Last 48 data points
        
        fig = go.Figure()
        
        # Add throughput rate line
        fig.add_trace(go.Scatter(
            x=trend_df['timestamp'],
            y=trend_df['throughput_rate'],
            mode='lines+markers',
            name='Throughput Rate',
            line=dict(color='#4472C4', width=3),
            hovertemplate='Time: %{x}<br>Throughput: %{y:.1f} UPH<extra></extra>'
        ))
        
        # Add target line
        fig.add_hline(
            y=TARGETS['overall_uph'], 
            line_dash="dash", 
            line_color="#28a745",
            annotation_text=f"Target: {TARGETS['overall_uph']} UPH"
        )
        
        fig.update_layout(
            title="Real-Time Throughput Performance",
            xaxis_title="Time",
            yaxis_title="Units Per Hour (UPH)",
            height=400,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("⏳ Collecting trend data... Refresh in a few minutes to see trends.")

# --- UPDATED CLEAN WORKSTATION FUNCTIONS ---
def create_line1_page(df):
    """Clean Line 1 page matching Power BI card style"""
    current_shift = get_current_shift()
    central_now = get_central_time_now()
    shift_start_time, _ = get_shift_window()
    hours_elapsed = max((central_now.replace(tzinfo=None) - shift_start_time.replace(tzinfo=None)).total_seconds() / 3600, 0.1)
    
    line1_stations = [f"L01OPS.IR2.{i}" for i in range(1, 27)]
    line1_data = df[df['workstation'].isin(line1_stations)] if 'workstation' in df.columns else pd.DataFrame()
    
    # Calculate line metrics
    total_units = len(line1_data)
    line_uph = total_units / hours_elapsed if hours_elapsed > 0 else 0
    active_stations = line1_data['workstation'].nunique() if not line1_data.empty else 0
    total_stations = len(line1_stations)
    
    # Status indicator
    status = "Critical" if line_uph < 30 else "Warning" if line_uph < 60 else "Good"
    status_color = "#dc3545" if status == "Critical" else "#fd7e14" if status == "Warning" else "#28a745"
    
    # Clean header matching Power BI
    col1, col2 = st.columns([4, 1])
    with col1:
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #4472C4 0%, #2E5984 100%); color: white; 
                    padding: 20px; border-radius: 8px; margin-bottom: 20px;
                    display: flex; justify-content: space-between; align-items: center;'>
            <div>
                <h2 style='margin: 0; font-size: 24px; font-weight: bold;'>RECEIVING LINE 1 HEALTH</h2>
                <p style='margin: 5px 0 0 0; font-size: 14px; opacity: 0.9;'>{current_shift} | Live Status</p>
            </div>
            <div style='display: flex; gap: 40px; align-items: center;'>
                <div style='text-align: center;'>
                    <div style='font-size: 32px; font-weight: bold; line-height: 1;'>{total_units}</div>
                    <div style='font-size: 12px; opacity: 0.8;'>Total Units</div>
                </div>
                <div style='text-align: center;'>
                    <div style='font-size: 32px; font-weight: bold; line-height: 1;'>{line_uph:.1f}</div>
                    <div style='font-size: 12px; opacity: 0.8;'>Line UPH</div>
                </div>
                <div style='text-align: center;'>
                    <div style='font-size: 32px; font-weight: bold; line-height: 1;'>{active_stations}/{total_stations}</div>
                    <div style='font-size: 12px; opacity: 0.8;'>Active Stations</div>
                </div>
            </div>
            <div style='background: {status_color}; color: white; padding: 8px 16px; 
                        border-radius: 20px; font-weight: bold; font-size: 14px;'>
                {status}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(create_live_status_indicator(), unsafe_allow_html=True)
    
    # Clean workstation grid
    create_power_bi_workstation_grid(df, line1_stations, "LINE 1", reverse_order=True)

def create_line2_page(df):
    """Clean Line 2 page matching Power BI card style"""
    current_shift = get_current_shift()
    central_now = get_central_time_now()
    shift_start_time, _ = get_shift_window()
    hours_elapsed = max((central_now.replace(tzinfo=None) - shift_start_time.replace(tzinfo=None)).total_seconds() / 3600, 0.1)
    
    line2_stations = [f"L01OPS.IR2.{i}" for i in range(27, 63)]
    line2_data = df[df['workstation'].isin(line2_stations)] if 'workstation' in df.columns else pd.DataFrame()
    
    # Calculate line metrics
    total_units = len(line2_data)
    line_uph = total_units / hours_elapsed if hours_elapsed > 0 else 0
    active_stations = line2_data['workstation'].nunique() if not line2_data.empty else 0
    total_stations = len(line2_stations)
    
    # Status indicator
    status = "Critical" if line_uph < 15 else "Warning" if line_uph < 30 else "Good"
    status_color = "#dc3545" if status == "Critical" else "#fd7e14" if status == "Warning" else "#28a745"
    
    # Clean header matching Power BI
    col1, col2 = st.columns([4, 1])
    with col1:
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #4472C4 0%, #2E5984 100%); color: white; 
                    padding: 20px; border-radius: 8px; margin-bottom: 20px;
                    display: flex; justify-content: space-between; align-items: center;'>
            <div>
                <h2 style='margin: 0; font-size: 24px; font-weight: bold;'>RECEIVING LINE 2 HEALTH</h2>
                <p style='margin: 5px 0 0 0; font-size: 14px; opacity: 0.9;'>{current_shift} | Live Status</p>
            </div>
            <div style='display: flex; gap: 40px; align-items: center;'>
                <div style='text-align: center;'>
                    <div style='font-size: 32px; font-weight: bold; line-height: 1;'>{total_units}</div>
                    <div style='font-size: 12px; opacity: 0.8;'>Total Units</div>
                </div>
                <div style='text-align: center;'>
                    <div style='font-size: 32px; font-weight: bold; line-height: 1;'>{line_uph:.1f}</div>
                    <div style='font-size: 12px; opacity: 0.8;'>Line UPH</div>
                </div>
                <div style='text-align: center;'>
                    <div style='font-size: 32px; font-weight: bold; line-height: 1;'>{active_stations}/{total_stations}</div>
                    <div style='font-size: 12px; opacity: 0.8;'>Active Stations</div>
                </div>
            </div>
            <div style='background: {status_color}; color: white; padding: 8px 16px; 
                        border-radius: 20px; font-weight: bold; font-size: 14px;'>
                {status}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(create_live_status_indicator(), unsafe_allow_html=True)
    
    # Clean workstation grid
    create_power_bi_workstation_grid(df, line2_stations, "LINE 2", reverse_order=False)

# --- POWER BI STYLE WORKSTATION GRID ---
# --- SIMPLIFIED POWER BI STYLE WORKSTATION GRID ---
# --- BULLETPROOF WORKSTATION GRID ---
# --- PROPER WORKSTATION GRID WITH PHYSICAL LINES ---
def create_power_bi_workstation_grid(df, stations, line_name, reverse_order=False):
    """Create workstation grid showing physical line layout with proper calculations"""

    if df.empty or 'workstation' not in df.columns:
        st.warning(f"No workstation data available for {line_name}")
        return

    central_now = get_central_time_now()
    station_data = {}

    # Process each station
    for station in stations:
        try:
            station_df = df[df['workstation'] == station]
            station_num = int(station.split('.')[-1])

            if not station_df.empty:
                imei_count = len(station_df)

                # Get operator
                operator = "Unknown"
                if 'user_description_filled' in station_df.columns:
                    latest_entry = station_df.iloc[-1]
                    operator = str(latest_entry.get('user_description_filled', 'Unknown')).strip()
                    if len(operator) > 8:
                        operator = operator[:8] + "..."

                # Calculate time active
                first_scan = pd.to_datetime(station_df['completion_date']).min()
                if pd.notna(first_scan):
                    time_diff = central_now.replace(tzinfo=None) - first_scan
                    hours_active = max(time_diff.total_seconds() / 3600, 0.1)
                    minutes_active = int(time_diff.total_seconds() / 60)

                    time_active = f"{minutes_active}m" if minutes_active < 60 else f"{minutes_active // 60}h {minutes_active % 60}m"
                    uph = round(imei_count / hours_active, 1)
                else:
                    time_active = "0m"
                    uph = 0.0

                station_data[station_num] = {
                    'station_name': station,
                    'imei_count': imei_count,
                    'uph': uph,
                    'time_active': time_active,
                    'operator': operator,
                    'active': True
                }
            else:
                station_data[station_num] = {
                    'station_name': station,
                    'imei_count': 0,
                    'uph': 0.0,
                    'time_active': "0m",
                    'operator': "",
                    'active': False
                }
        except Exception:
            station_data[int(station.split('.')[-1])] = {
                'station_name': station,
                'imei_count': 0,
                'uph': 0.0,
                'time_active': "0m",
                'operator': "Error",
                'active': False
            }

    # Group stations by line
    line1_stations = [int(s.split('.')[-1]) for s in stations if 1 <= int(s.split('.')[-1]) <= 26]
    line2_stations = [int(s.split('.')[-1]) for s in stations if 27 <= int(s.split('.')[-1]) <= 62]

    # Display layout
    if line_name == "LINE 1":
        st.subheader("🏭 Receiving Line 1 - Physical Layout")
        if line1_stations:
            st.markdown("**Physical Line 1 (Stations 1–26)**")
            create_station_row(station_data, line1_stations, reverse_order)

    elif line_name == "LINE 2":
        st.subheader("🏭 Receiving Line 2 - Physical Layout")
        if line2_stations:
            st.markdown("**Physical Line 2 (Stations 27–62)**")
            create_station_row(station_data, line2_stations, reverse_order)


def create_station_row(station_data, station_nums, reverse_order=False, columns_per_row=13):
    """Create a row of station cards using Streamlit columns"""

    sorted_stations = sorted(station_nums, reverse=reverse_order)

    for i in range(0, len(sorted_stations), columns_per_row):
        cols = st.columns(columns_per_row)
        for j, station_num in enumerate(sorted_stations[i:i + columns_per_row]):
            if station_num in station_data:
                info = station_data[station_num]

                # Determine card colors
                if info['active']:
                    if info['imei_count'] > 5:
                        bg_color = "#d4edda"
                        border_color = "#28a745"
                    elif info['imei_count'] > 0:
                        bg_color = "#fff3cd"
                        border_color = "#ffc107"
                    else:
                        bg_color = "#f8d7da"
                        border_color = "#dc3545"
                else:
                    bg_color = "#f8f9fa"
                    border_color = "#6c757d"

                with cols[j]:
                    st.markdown(f"""
                    <div style='
                        border: 2px solid {border_color};
                        border-radius: 2px;
                        padding: 4px;
                        background: {bg_color};
                        text-align: center;
                        font-family: Arial, sans-serif;
                        font-size: 9px;
                        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
                        width: 110px;
                        min-height: 130px;
                    '>
                        <div style='font-weight: bold; color: #495057; font-size: 12px; margin-bottom: 2px;'>
                            {info['station_name']}
                        </div>
                        <div style='font-size: 25px; font-weight: bold; color: #212529; margin: 1px 0;'>
                            {info['imei_count']}
                        </div>
                        <div style='color: #6c757d; font-size: 15px; margin-bottom: 1px;'>IMEIs</div>
                        <div style='color: #495057; font-size: 17px; margin-bottom: 1px;'>
                            UPH: {info['uph']}
                        </div>
                        <div style='color: #495057; font-size: 12px; margin-bottom: 1px;'>
                            Active: {info['time_active']}
                        </div>
                        <div style='color: #495057; font-size: 15px; background: rgba(0,0,0,0.05); 
                                    padding: 1px 2px; border-radius: 2px; overflow: hidden; text-overflow: ellipsis;'>
                            {info['operator'] if info['operator'] else 'Inactive'}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

    
    # Show summary for this row
    active_count = sum(1 for num in sorted_stations if station_data[num]['active'])
    total_imeis = sum(station_data[num]['imei_count'] for num in sorted_stations if num in station_data)
    avg_uph = sum(station_data[num]['uph'] for num in sorted_stations if num in station_data and station_data[num]['active'])
    avg_uph = avg_uph / max(active_count, 1)
    
    st.info(f"📊 {active_count}/{len(sorted_stations)} active | {total_imeis} total IMEIs | {avg_uph:.1f} avg UPH")
 


# --- UPDATED PROGRAMS PAGE ---
def create_programs_page(df):
    """Clean programs page matching Power BI card style"""
    current_shift = get_current_shift()
    
    # Clean header with live indicator
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #4472C4 0%, #2E5984 100%); color: white; 
                    padding: 15px; border-radius: 8px; margin-bottom: 20px; text-align: center;'>
            <h2 style='margin: 0; font-size: 22px; font-weight: bold;'>📱 Program Distribution</h2>
            <p style='margin: 5px 0 0 0; font-size: 14px; opacity: 0.9;'>{current_shift} | Live Status</p>
        </div>
        """, unsafe_allow_html=True)
        
        if df.empty:
            st.warning("No program data available for the current filters.")
            return
    
    with col2:
        st.markdown(create_live_status_indicator(), unsafe_allow_html=True)
    
    # Process program data
    program_summary = df.groupby('program_description').agg({'imei': 'nunique'}).reset_index()
    program_summary.columns = ['program', 'imei_count']
    program_summary = program_summary.sort_values('imei_count', ascending=False)
    total_devices = program_summary['imei_count'].sum()
    
    # Overall summary card
    st.markdown(f"""
    <div style='background: #e3f2fd; padding: 15px; border-radius: 8px; margin-bottom: 20px; 
                border-left: 4px solid #4472C4; text-align: center;'>
        <h3 style='margin: 0; color: #1976d2;'>📊 Total Devices Processed: {total_devices:,}</h3>
        <p style='margin: 5px 0 0 0; color: #666; font-size: 14px;'>Across {len(program_summary)} Programs</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create Power BI style program cards
    if not program_summary.empty:
        # Arrange in 4 columns for cleaner layout
        num_cols = 4
        
        # Group programs into rows of 4
        for row_start in range(0, len(program_summary), num_cols):
            cols = st.columns(num_cols)
            row_programs = program_summary.iloc[row_start:row_start + num_cols]
            
            for i, (_, program) in enumerate(row_programs.iterrows()):
                with cols[i]:
                    percentage = (program['imei_count'] / total_devices * 100) if total_devices > 0 else 0
                    
                    # Color scheme based on volume
                    if percentage >= 20:
                        card_color = "#4472C4"  # Blue for high volume
                        text_color = "white"
                    elif percentage >= 10:
                        card_color = "#28a745"  # Green for medium volume
                        text_color = "white"
                    elif percentage >= 5:
                        card_color = "#fd7e14"  # Orange for low volume
                        text_color = "white"
                    else:
                        card_color = "#6c757d"  # Gray for very low volume
                        text_color = "white"
                    
                    # Create Power BI style card
                    st.markdown(f"""
                    <div style='
                        background: {card_color}; 
                        color: {text_color}; 
                        padding: 20px; 
                        border-radius: 8px; 
                        text-align: center; 
                        margin-bottom: 15px;
                        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                        min-height: 130px;
                        display: flex;
                        flex-direction: column;
                        justify-content: space-between;
                    '>
                        <div style='font-size: 11px; font-weight: bold; opacity: 0.9; margin-bottom: 8px; 
                                    overflow: hidden; text-overflow: ellipsis; white-space: nowrap;'>
                            {program['program']}
                        </div>
                        <div style='font-size: 36px; font-weight: bold; line-height: 1; margin: 10px 0;'>
                            {program['imei_count']:,}
                        </div>
                        <div style='font-size: 10px; opacity: 0.8; margin-bottom: 8px;'>DEVICES</div>
                        <div style='background: rgba(255,255,255,0.2); padding: 4px 8px; border-radius: 12px; 
                                    font-size: 12px; font-weight: bold;'>
                            {percentage:.1f}%
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
    
    # Add summary statistics at the bottom
    if not program_summary.empty:
        st.divider()
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Top Program", program_summary.iloc[0]['program'][:15] + "..." if len(program_summary.iloc[0]['program']) > 15 else program_summary.iloc[0]['program'], 
                     f"{program_summary.iloc[0]['imei_count']:,} devices")
        
        with col2:
            avg_devices = program_summary['imei_count'].mean()
            st.metric("Average per Program", f"{avg_devices:.0f}", "devices")
        
        with col3:
            top_3_percentage = program_summary.head(3)['imei_count'].sum() / total_devices * 100
            st.metric("Top 3 Programs", f"{top_3_percentage:.1f}%", "of total volume")
        
        with col4:
            programs_over_5_percent = len(program_summary[program_summary['imei_count'] / total_devices >= 0.05])
            st.metric("Major Programs", programs_over_5_percent, "≥5% volume")

# --- Other Page Functions (Attendance, Unknowns, etc.) ---
def create_attendance_page(df):
    """Enhanced attendance page with live indicators"""
    current_shift = get_current_shift()
    central_now = get_central_time_now()
    receiving_workers = df[
        (df['user_description_filled'].notna()) &
        (df['workstation'].notna()) &
        (df['workstation'].str.contains('RCV|IR2|RECEIVING', case=False, na=False))
    ].copy() if not df.empty else pd.DataFrame()
    total_operators = receiving_workers['user_description_filled'].nunique()

    # Header with live indicator
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(f"""<div style='background: linear-gradient(135deg, #4472C4 0%, #2E5984 100%); color: white; text-align: center;
                    padding: 12px; font-weight: bold; font-size: 16px; margin-bottom: 15px; border-radius: 8px;'>
            Receiving Area Attendance - {current_shift} | {total_operators} Operators Active
        </div>""", unsafe_allow_html=True)
    
    with col2:
        st.markdown(create_live_status_indicator(), unsafe_allow_html=True)
    
    if receiving_workers.empty:
        st.info(f"No operators currently active in Receiving Area for {current_shift}")
        return
    
    user_summary = receiving_workers.groupby('user_description_filled').agg({'workstation': 'last','completion_date': 'max'}).reset_index()
    try:
        user_summary['station_number'] = user_summary['workstation'].str.extract(r'\.(\d+)').astype(float).fillna(0).astype(int)
    except:
        user_summary['station_number'] = 0
    user_summary = user_summary.sort_values(by='station_number').reset_index(drop=True)
    
    cols = st.columns(7)
    for i, row in user_summary.iterrows():
        with cols[i % 7]:
            last_active_time = pd.to_datetime(row['completion_date'])
            minutes_ago = int((central_now.replace(tzinfo=None) - last_active_time).total_seconds() / 60)
            time_str = f"{minutes_ago}m ago" if minutes_ago < 60 else f"{minutes_ago // 60}h ago"
            if minutes_ago <= 30: 
                border_color = "#4CAF50"
            else: 
                border_color = "#FF9800"
            st.markdown(f"""<div style='background: white; border: 2px solid {border_color}; padding: 1vh; text-align: center; flex: 1 1 auto; min-width: 8vw; max-width: 15vw; min-height: 8vh; max-height: 12vh; margin: 0.3vh; border-radius: 0.5vh; box-shadow: 0 0.2vh 0.4vh rgba(0,0,0,0.1); display: flex; flex-direction: column; justify-content: space-between;'>
                <div style='font-weight: bold; color: #4472C4; font-size: clamp(8px, 1.2vw, 14px); line-height: 1.1; margin-bottom: 0.5vh; flex: 1; display: flex; align-items: center; justify-content: center; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;'>{row['user_description_filled']}</div>
                <div style='background: #E3F2FD; color: #1976D2; font-size: clamp(6px, 0.9vw, 13px); font-weight: bold; padding: 0.3vh 0.6vw; border-radius: 0.3vh; border: 1px solid #90CAF9; margin-bottom: 0.5vh; flex: 1; display: flex; align-items: center; justify-content: center; word-break: break-all; line-height: 1;'>{row['workstation']}</div>
                <div style='background: {border_color}; color: white; font-size: clamp(6px, 0.8vw, 12px); font-weight: bold; padding: 0.4vh 1vw; border-radius: 1vh; display: flex; align-items: center; justify-content: center;'>ACTIVE ({time_str})</div>
            </div>""", unsafe_allow_html=True)

def create_unknowns_page(df_unknown):
    """Creates the unknowns page with proper Streamlit components"""
    current_shift = get_current_shift()
    
    # Header with live indicator
    col1, col2 = st.columns([3, 1])
    with col1:
        st.subheader(f"❓ Unknown Hold - {current_shift}")
        df_unique_unknowns = df_unknown.drop_duplicates(subset=['imei'], keep='first')
        if df_unique_unknowns.empty:
            st.success("No devices in Unknown Hold")
        else:
            total_unknown = len(df_unique_unknowns)
            st.error(f"Total Unknown Devices: {total_unknown:,}")
            st.dataframe(df_unique_unknowns, use_container_width=True)
    
    with col2:
        st.markdown(create_live_status_indicator(), unsafe_allow_html=True)

def create_unknown_rework_page(df_unknown):
    """Creates the unknown rework page with proper Streamlit components"""
    current_shift = get_current_shift()
    
    # Header with live indicator
    col1, col2 = st.columns([3, 1])
    with col1:
        st.subheader(f"🔄 Unknown Hold Rework - {current_shift}")
        if df_unknown.empty:
            rework_items = pd.DataFrame()
        else:
            rework_items = df_unknown[df_unknown['recovery_status'].isin(['N', 'MC'])]
        if rework_items.empty:
            st.success(f"No Unknown Hold rework items for {current_shift}")
        else:
            total_rework = len(rework_items)
            st.warning(f"Rework Required: {total_rework:,} IMEIs Need Action")
            st.dataframe(rework_items, use_container_width=True)
    
    with col2:
        st.markdown(create_live_status_indicator(), unsafe_allow_html=True)

def create_raw_data_page(df):
    """Creates a raw data page with export functionality and proper Streamlit components"""
    # Header with live indicator
    col1, col2 = st.columns([3, 1])
    with col1:
        st.subheader("📊 Raw Data")
        st.info(f"Displaying {len(df):,} records based on current filters.")
        st.download_button("Export to CSV", df.to_csv().encode('utf-8'), "manufacturing_data.csv", "text/csv")
        st.dataframe(df, use_container_width=True)
    
    with col2:
        st.markdown(create_live_status_indicator(), unsafe_allow_html=True)

def create_filter_page(df_master):
    """Creates a dedicated page for advanced filters with proper Streamlit components"""
    # Header with live indicator
    col1, col2 = st.columns([3, 1])
    with col1:
        st.subheader("⚙️ Advanced Filter Control Panel")
        st.info("Select filters below. The dashboard will automatically update on the next data refresh.")
    
    with col2:
        st.markdown(create_live_status_indicator(), unsafe_allow_html=True)
    
    # --- Multi-Select Filter Widgets ---
    col1, col2 = st.columns(2)
    with col1:
        programs_options = sorted(df_master['program_description'].dropna().unique().tolist())
        st.session_state.filter_program = st.multiselect("Program(s)", programs_options, default=st.session_state.get('filter_program', []))
        
        operations_options = sorted(df_master['source'].dropna().unique().tolist())
        st.session_state.filter_operation = st.multiselect("Operation(s)", operations_options, default=st.session_state.get('filter_operation', []))
        
    with col2:
        users_options = sorted(df_master['user_description_filled'].dropna().unique().tolist())
        st.session_state.filter_user = st.multiselect("User(s)", users_options, default=st.session_state.get('filter_user', []))
        
        st.session_state.filter_line = st.multiselect("Production Line(s)", ['Line 1 Only', 'Line 2 Only'], default=st.session_state.get('filter_line', []))
    
    st.session_state.filter_min_uph = st.number_input("Minimum UPH", min_value=0.0, value=st.session_state.get('filter_min_uph', 0.0), step=1.0)
    st.session_state.filter_show_inactive = st.checkbox("Show Inactive Stations", value=st.session_state.get('filter_show_inactive', True))

    if st.button("Apply Filters", type="primary"):
        st.success("Filters applied! Returning to dashboard.")
        st.session_state.selected_page = "📊 KPI Dashboard"
        safe_rerun()

# --- Persistent Filter System ---
def apply_persistent_filters(df_master):
    """Apply the persistent filters to the dataframe based on session state."""
    filtered_df = df_master.copy()
    
    if st.session_state.get('filter_program') and st.session_state.filter_program:
        filtered_df = filtered_df[filtered_df['program_description'].isin(st.session_state.filter_program)]
        
    if st.session_state.get('filter_line') and st.session_state.filter_line:
        all_stations = []
        if 'Line 1 Only' in st.session_state.filter_line:
            all_stations.extend([f"L01OPS.IR2.{i}" for i in range(1, 27)])
        if 'Line 2 Only' in st.session_state.filter_line:
            all_stations.extend([f"L01OPS.IR2.{i}" for i in range(27, 63)])
        filtered_df = filtered_df[filtered_df['workstation'].isin(all_stations)]
        
    if st.session_state.get('filter_user') and st.session_state.filter_user:
        filtered_df = filtered_df[filtered_df['user_description_filled'].isin(st.session_state.filter_user)]
        
    if st.session_state.get('filter_operation') and st.session_state.filter_operation:
        filtered_df = filtered_df[filtered_df['source'].isin(st.session_state.filter_operation)]
        
    return filtered_df

# --- Utility Functions ---
def safe_rerun():
    """Safe page rerun with state preservation"""
    try:
        st.rerun()  # Use st.rerun() instead of st.experimental_rerun()
    except Exception:
        # Fallback for older Streamlit versions
        st.experimental_rerun()

def setup_auto_refresh():
    """Proper auto-refresh implementation with unique key"""
    if st.sidebar.checkbox("🔄 Auto-refresh (5 min)", key="sidebar_auto_refresh"):
        # Check if 5 minutes have passed
        if (datetime.now() - st.session_state.last_refresh).total_seconds() > 300:
            st.cache_data.clear()
            st.session_state.last_refresh = datetime.now()
            safe_rerun()

def cleanup_dataframes():
    """Clean up large dataframes from memory"""
    if 'large_dataframes' in st.session_state:
        for df_name in st.session_state.large_dataframes:
            if df_name in st.session_state:
                del st.session_state[df_name]
        st.session_state.large_dataframes = []

# --- Main Application Logic ---
def main():
    """Enhanced main function with better error handling and performance"""
    try:
        # Initialize with robust error handling
        initialize_session_state()
        
        # Setup auto-refresh
        setup_auto_refresh()
        
        # Sidebar with error handling
        with st.sidebar:
            st.markdown("## 🧭 Navigation")
            
            # Performance monitoring
            if st.session_state.get('error_count', 0) > 5:
                st.warning("⚠️ Multiple errors detected. Consider refreshing.")
                if st.button("🔄 Reset Session"):
                    for key in list(st.session_state.keys()):
                        del st.session_state[key]
                    safe_rerun()
            
            pages = {
                "📊 KPI Dashboard": create_enhanced_kpi_overview_page,
                "🏭 Line 1 Workstations": create_line1_page,
                "🏭 Line 2 Workstations": create_line2_page,
                "👥 Attendance": create_attendance_page,
                "📱 Programs": create_programs_page,
                "❓ Unknowns": create_unknowns_page,
                "🔄 Unknown Rework": create_unknown_rework_page,
                "📋 Raw Data": create_raw_data_page,
                "⚙️ Advanced Filters": create_filter_page
            }
            
            st.session_state.selected_page = st.radio(
                "Go to:", 
                list(pages.keys()), 
                key="page_selector",
                index=list(pages.keys()).index(st.session_state.get('selected_page', '📊 KPI Dashboard'))
            )
            
            st.divider()
            
            # Manual refresh with loading state
            if st.button("🔄 Refresh Data", use_container_width=True):
                with st.spinner("Refreshing data..."):
                    st.cache_data.clear()
                    cleanup_dataframes()
                    safe_rerun()
            
            # Display connection status and live indicator
            col1, col2 = st.columns(2)
            with col1:
                if st.session_state.get('data_load_error'):
                    st.error("❌ Connection Issues")
                else:
                    st.success("✅ Connected")
            
            with col2:
                st.markdown(create_live_status_indicator(), unsafe_allow_html=True)
            
            st.caption(f"Last refresh: {st.session_state.last_refresh.strftime('%I:%M:%S %p')}")
            
            # Quick stats in sidebar
            if st.session_state.get('cached_data'):
                dfs, _, _ = st.session_state.cached_data
                if dfs and not dfs.get('item_receipt', pd.DataFrame()).empty:
                    total_processed = len(dfs.get('receipt_complete', pd.DataFrame()))
                    total_unknown = len(dfs.get('unknown_analysis', pd.DataFrame()))
                    st.markdown("### 📈 Quick Stats")
                    st.metric("Completed", total_processed)
                    st.metric("Unknown Hold", total_unknown)
        
        # Load data with robust error handling
        try:
            dfs, start_time, end_time = load_data()
        except Exception as e:
            st.error(f"Critical data loading error: {e}")
            logger.error(f"Critical data loading error: {e}")
            st.stop()
        
        if not dfs or st.session_state.data_load_error:
            st.error("Unable to load data. Please check your database connection.")
            st.stop()
        
        # Process data with error handling
        try:
            df_all = optimized_data_processing(dfs)
            df_unknown = dfs.get("unknown_analysis", pd.DataFrame())
            df_all = add_unknown_rework_column(df_all, df_unknown)
            df_all = fill_missing_user_descriptions(df_all)
            df_master = rank_latest_imei(df_all)
        except Exception as e:
            st.error(f"Data processing error: {e}")
            logger.error(f"Data processing error: {e}")
            st.session_state.error_count = st.session_state.get('error_count', 0) + 1
            st.stop()
        
        # Route to appropriate page with error handling
        try:
            if st.session_state.selected_page == "⚙️ Advanced Filters":
                pages["⚙️ Advanced Filters"](df_master)
            else:
                filtered_df = apply_persistent_filters(df_master)
                selected_page_func = pages[st.session_state.selected_page]
                
                if selected_page_func == create_enhanced_kpi_overview_page:
                    selected_page_func(dfs, filtered_df)
                elif selected_page_func in [create_unknowns_page, create_unknown_rework_page]:
                    selected_page_func(df_unknown)
                else:
                    selected_page_func(filtered_df)
        except Exception as e:
            st.error(f"Page rendering error: {e}")
            logger.error(f"Page rendering error: {e}")
            st.session_state.error_count = st.session_state.get('error_count', 0) + 1
            
    except Exception as e:
        st.error(f"Critical application error: {e}")
        logger.error(f"Critical error in main: {e}")

if __name__ == "__main__":
    main()
