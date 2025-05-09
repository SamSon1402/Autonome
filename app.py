import streamlit as st
import pandas as pd
import numpy as np
import datetime
import os
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# Set page configuration
st.set_page_config(
    page_title="Project Synapse Core",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Function to configure plots for dark theme - defined at the top level
def configure_plot_for_dark_theme(fig, ax):
    """Configure matplotlib plots for dark theme"""
    fig.patch.set_facecolor('#000000')
    ax.set_facecolor('#000000')
    ax.tick_params(colors='white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.title.set_color('white')
    for spine in ax.spines.values():
        spine.set_edgecolor('white')
    return fig, ax

# Custom CSS for retro gaming aesthetic
def load_css():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=VT323&family=Space+Mono&display=swap');
        
        /* Main theme */
        :root {
            --main-bg-color: black;
            --main-text-color: white;
            --inverse-bg-color: white;
            --inverse-text-color: black;
            --border-color: white;
            --pixel-size: 2px;
        }
        
        /* Global styles */
        html, body, [class*="css"] {
            font-family: 'VT323', monospace !important;
            letter-spacing: 1px;
            color: var(--main-text-color);
            background-color: var(--main-bg-color);
        }
        
        /* Headers */
        h1, h2, h3, h4, h5, h6 {
            font-family: 'VT323', monospace !important;
            text-transform: uppercase;
            letter-spacing: 2px;
        }
        
        /* Pixel borders */
        .pixel-border {
            border: var(--pixel-size) solid var(--border-color);
            box-shadow: var(--pixel-size) var(--pixel-size) 0 0 var(--border-color);
        }
        
        /* Container styling */
        .block-container {
            padding: 2rem;
        }
        
        /* Streamlit Components Styling */
        .stButton button {
            font-family: 'VT323', monospace !important;
            border: var(--pixel-size) solid var(--border-color) !important;
            box-shadow: var(--pixel-size) var(--pixel-size) 0 0 var(--border-color) !important;
            background-color: var(--main-bg-color) !important;
            color: var(--main-text-color) !important;
            border-radius: 0 !important;
            padding: 0.5rem 1rem !important;
            transition: transform 0.1s !important;
        }
        
        .stButton button:active {
            transform: translate(var(--pixel-size), var(--pixel-size)) !important;
            box-shadow: 0 0 0 0 var(--border-color) !important;
        }
        
        /* Sidebar */
        .css-1d391kg {
            background-color: var(--inverse-bg-color) !important;
        }
        
        .css-1d391kg p, .css-1d391kg h1, .css-1d391kg h2, .css-1d391kg h3, .css-1d391kg h4 {
            color: var(--inverse-text-color) !important;
        }
        
        /* Input fields */
        input, select, textarea {
            font-family: 'Space Mono', monospace !important;
            border: var(--pixel-size) solid var(--border-color) !important;
            border-radius: 0 !important;
        }
        
        /* Sliders */
        .stSlider div[data-baseweb="slider"] {
            height: 20px !important;
        }
        
        /* Health bar styling */
        .health-bar {
            width: 100%;
            height: 20px;
            border: var(--pixel-size) solid var(--border-color);
            position: relative;
            margin: 10px 0;
        }
        
        .health-bar-inner {
            height: 100%;
            background-color: var(--inverse-bg-color);
        }
        
        /* Code blocks */
        code {
            font-family: 'Space Mono', monospace !important;
            background-color: var(--inverse-bg-color) !important;
            color: var(--inverse-text-color) !important;
            border-radius: 0 !important;
            padding: 2px 4px !important;
        }
        
        /* Scanline effect */
        .scanlines {
            position: relative;
        }
        
        .scanlines::before {
            content: "";
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: linear-gradient(
                to bottom,
                transparent 50%,
                rgba(0, 0, 0, 0.1) 50%
            );
            background-size: 100% 4px;
            pointer-events: none;
            z-index: 1;
        }
        
        /* Custom Tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 2px;
        }
        
        .stTabs [data-baseweb="tab"] {
            font-family: 'VT323', monospace !important;
            border: var(--pixel-size) solid var(--border-color) !important;
            border-radius: 0 !important;
            padding: 0.5rem 1rem !important;
            background-color: var(--main-bg-color) !important;
            color: var(--main-text-color) !important;
        }
        
        .stTabs [aria-selected="true"] {
            background-color: var(--inverse-bg-color) !important;
            color: var(--inverse-text-color) !important;
        }
    </style>
    """, unsafe_allow_html=True)

load_css()

# Initialize session state
if 'user_data' not in st.session_state:
    st.session_state.user_data = pd.DataFrame()
if 'page' not in st.session_state:
    st.session_state.page = 'dashboard'
if 'current_user' not in st.session_state:
    st.session_state.current_user = None

# Data file path
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# Functions for data handling
def save_data(data, username):
    """Save user data to CSV."""
    filepath = os.path.join(DATA_DIR, f"{username}.csv")
    data.to_csv(filepath, index=False)

def load_data(username):
    """Load user data from CSV."""
    filepath = os.path.join(DATA_DIR, f"{username}.csv")
    if os.path.exists(filepath):
        return pd.read_csv(filepath)
    else:
        return pd.DataFrame(columns=['date', 'mood', 'stress', 'sleep_hours', 'activity_minutes', 'symptoms'])

def add_entry(data, entry):
    """Add a new entry to the user data."""
    return pd.concat([data, pd.DataFrame([entry])], ignore_index=True)

# Functions for insights
def generate_trend_insight(data):
    """Generate simple insights about trends in the data."""
    if len(data) < 5:
        return "Need more data to generate insights."
    
    insights = []
    
    # Check for mood-sleep correlation
    if 'mood' in data.columns and 'sleep_hours' in data.columns:
        correlation = data['mood'].corr(data['sleep_hours'])
        if abs(correlation) > 0.5:
            direction = "positively" if correlation > 0 else "negatively"
            insights.append(f"Your mood appears to be {direction} correlated with your sleep.")
    
    # Check for stress-activity correlation
    if 'stress' in data.columns and 'activity_minutes' in data.columns:
        correlation = data['stress'].corr(data['activity_minutes'])
        if abs(correlation) > 0.5:
            direction = "increase" if correlation > 0 else "decrease"
            insights.append(f"Your stress levels tend to {direction} with more physical activity.")
    
    # Check for recent mood trends
    if 'mood' in data.columns and len(data) >= 7:
        recent_mood = data['mood'].iloc[-7:].mean()
        overall_mood = data['mood'].mean()
        if recent_mood - overall_mood > 0.5:
            insights.append("Your mood has been better than usual in the past week.")
        elif overall_mood - recent_mood > 0.5:
            insights.append("Your mood has been lower than usual in the past week.")
    
    return insights if insights else "No significant trends detected yet."

def detect_anomalies(data):
    """Detect simple anomalies in the user data."""
    if len(data) < 7:
        return "Need more data to detect anomalies."
    
    anomalies = []
    
    # Check for low sleep streak
    if 'sleep_hours' in data.columns:
        recent_sleep = data['sleep_hours'].iloc[-3:].values
        if all(hours < 6 for hours in recent_sleep):
            anomalies.append("You've had consistently low sleep for the past 3 days.")
    
    # Check for high stress streak
    if 'stress' in data.columns:
        recent_stress = data['stress'].iloc[-3:].values
        if all(level > 7 for level in recent_stress):
            anomalies.append("Your stress levels have been high for the past 3 days.")
    
    # Check for sudden mood changes
    if 'mood' in data.columns and len(data) >= 2:
        recent_mood_change = abs(data['mood'].iloc[-1] - data['mood'].iloc[-2])
        if recent_mood_change >= 3:
            direction = "up" if data['mood'].iloc[-1] > data['mood'].iloc[-2] else "down"
            anomalies.append(f"Your mood changed significantly {direction} in your last entry.")
    
    return anomalies if anomalies else "No anomalies detected."

# Sidebar
with st.sidebar:
    st.markdown('<h1 style="text-align: center;">SYNAPSE CORE</h1>', unsafe_allow_html=True)
    st.markdown('<div class="scanlines"><h2 style="text-align: center;">Health_Tracker.exe</h2></div>', unsafe_allow_html=True)
    
    # Simple "login" system
    if st.session_state.current_user is None:
        username = st.text_input("Enter Your Username", key="username_input")
        if st.button("START"):
            if username:
                st.session_state.current_user = username
                st.session_state.user_data = load_data(username)
                st.rerun()
    else:
        st.markdown(f"<h3>PLAYER: {st.session_state.current_user}</h3>", unsafe_allow_html=True)
        
        # Navigation
        if st.button("DASHBOARD"):
            st.session_state.page = 'dashboard'
        
        if st.button("ADD HEALTH DATA"):
            st.session_state.page = 'add_data'
        
        if st.button("INSIGHTS"):
            st.session_state.page = 'insights'
        
        if st.button("SETTINGS"):
            st.session_state.page = 'settings'
        
        if st.button("LOGOUT"):
            st.session_state.current_user = None
            st.rerun()

# Main content
if st.session_state.current_user is None:
    # Welcome screen
    st.markdown("""
    <div style="text-align: center;">
        <h1 class="scanlines">PROJECT SYNAPSE CORE</h1>
        <h2>Your Personal Health Log & Insights</h2>
        <p style="font-size: 1.5rem;">Enter your username and press START to begin</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div style="text-align: center; padding: 2rem; border: 4px solid white; margin-top: 2rem;">
            <h3>HEALTH STATS TRACKING</h3>
            <p>➤ Monitor your mood and stress</p>
            <p>➤ Track sleep quality</p>
            <p>➤ Log physical activity</p>
            <p>➤ Record symptoms</p>
            <h3>AI-POWERED INSIGHTS</h3>
            <p>➤ Identify health trends</p>
            <p>➤ Detect pattern anomalies</p>
            <p>➤ Visualize your health journey</p>
        </div>
        """, unsafe_allow_html=True)

# Dashboard page
elif st.session_state.page == 'dashboard':
    st.markdown('<h1 class="scanlines">HEALTH DASHBOARD</h1>', unsafe_allow_html=True)
    
    if len(st.session_state.user_data) == 0:
        st.markdown("""
        <div style="text-align: center; padding: 2rem;">
            <h2>No data available</h2>
            <p>Add your first health entry to see your dashboard</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Recent stats
        st.markdown('<h2>RECENT STATS</h2>', unsafe_allow_html=True)
        
        cols = st.columns(4)
        
        recent_data = st.session_state.user_data.iloc[-1] if len(st.session_state.user_data) > 0 else None
        
        if recent_data is not None:
            # Mood meter
            with cols[0]:
                st.markdown('<h3>MOOD</h3>', unsafe_allow_html=True)
                mood = recent_data['mood'] if 'mood' in recent_data else 0
                mood_percentage = int((mood / 10) * 100)
                st.markdown(f"""
                <div class="health-bar">
                    <div class="health-bar-inner" style="width: {mood_percentage}%;"></div>
                </div>
                <p style="text-align: center;">{mood}/10</p>
                """, unsafe_allow_html=True)
            
            # Stress meter
            with cols[1]:
                st.markdown('<h3>STRESS</h3>', unsafe_allow_html=True)
                stress = recent_data['stress'] if 'stress' in recent_data else 0
                stress_percentage = int((stress / 10) * 100)
                st.markdown(f"""
                <div class="health-bar">
                    <div class="health-bar-inner" style="width: {stress_percentage}%;"></div>
                </div>
                <p style="text-align: center;">{stress}/10</p>
                """, unsafe_allow_html=True)
            
            # Sleep meter
            with cols[2]:
                st.markdown('<h3>SLEEP</h3>', unsafe_allow_html=True)
                sleep = recent_data['sleep_hours'] if 'sleep_hours' in recent_data else 0
                sleep_percentage = int((sleep / 12) * 100)
                st.markdown(f"""
                <div class="health-bar">
                    <div class="health-bar-inner" style="width: {sleep_percentage}%;"></div>
                </div>
                <p style="text-align: center;">{sleep} hours</p>
                """, unsafe_allow_html=True)
            
            # Activity meter
            with cols[3]:
                st.markdown('<h3>ACTIVITY</h3>', unsafe_allow_html=True)
                activity = recent_data['activity_minutes'] if 'activity_minutes' in recent_data else 0
                activity_percentage = int((activity / 120) * 100)
                st.markdown(f"""
                <div class="health-bar">
                    <div class="health-bar-inner" style="width: {activity_percentage}%;"></div>
                </div>
                <p style="text-align: center;">{activity} mins</p>
                """, unsafe_allow_html=True)
        
        # Charts
        st.markdown('<h2>HEALTH HISTORY</h2>', unsafe_allow_html=True)
        
        tab1, tab2, tab3, tab4 = st.tabs(["MOOD & STRESS", "SLEEP", "ACTIVITY", "SYMPTOMS"])
        
        with tab1:
            if 'mood' in st.session_state.user_data.columns and 'stress' in st.session_state.user_data.columns:
                # Create retro-style plot
                fig, ax = plt.subplots(figsize=(10, 5))
                fig, ax = configure_plot_for_dark_theme(fig, ax)
                
                # Convert date to datetime if it's not already
                if 'date' in st.session_state.user_data.columns:
                    dates = pd.to_datetime(st.session_state.user_data['date'])
                else:
                    dates = range(len(st.session_state.user_data))
                
                # Plot mood and stress with step-style lines for retro feel
                ax.step(dates, st.session_state.user_data['mood'], where='mid', label='Mood', color='white', linestyle='-', linewidth=2, marker='s')
                ax.step(dates, st.session_state.user_data['stress'], where='mid', label='Stress', color='white', linestyle='--', linewidth=2, marker='o')
                
                # Set y-axis limits
                ax.set_ylim(0, 11)
                
                # Add grid for retro feel
                ax.grid(True, linestyle='--', alpha=0.7)
                
                # Add legend
                ax.legend()
                
                # Set labels
                ax.set_ylabel('Level (0-10)')
                ax.set_title('Mood & Stress History')
                
                # Remove spines
                for spine in ['top', 'right']:
                    ax.spines[spine].set_visible(False)
                
                st.pyplot(fig)
            else:
                st.markdown('<p>No mood or stress data available</p>', unsafe_allow_html=True)
        
        with tab2:
            if 'sleep_hours' in st.session_state.user_data.columns:
                # Create retro-style plot
                fig, ax = plt.subplots(figsize=(10, 5))
                fig, ax = configure_plot_for_dark_theme(fig, ax)
                
                # Convert date to datetime if it's not already
                if 'date' in st.session_state.user_data.columns:
                    dates = pd.to_datetime(st.session_state.user_data['date'])
                else:
                    dates = range(len(st.session_state.user_data))
                
                # Plot sleep with bar style for retro feel
                ax.bar(dates, st.session_state.user_data['sleep_hours'], color='white', width=0.6)
                
                # Set y-axis limits
                ax.set_ylim(0, max(12, st.session_state.user_data['sleep_hours'].max() + 1))
                
                # Add grid for retro feel
                ax.grid(True, linestyle='--', alpha=0.7, axis='y')
                
                # Set labels
                ax.set_ylabel('Hours')
                ax.set_title('Sleep History')
                
                # Remove spines
                for spine in ['top', 'right']:
                    ax.spines[spine].set_visible(False)
                
                st.pyplot(fig)
            else:
                st.markdown('<p>No sleep data available</p>', unsafe_allow_html=True)
        
        with tab3:
            if 'activity_minutes' in st.session_state.user_data.columns:
                # Create retro-style plot
                fig, ax = plt.subplots(figsize=(10, 5))
                fig, ax = configure_plot_for_dark_theme(fig, ax)
                
                # Convert date to datetime if it's not already
                if 'date' in st.session_state.user_data.columns:
                    dates = pd.to_datetime(st.session_state.user_data['date'])
                else:
                    dates = range(len(st.session_state.user_data))
                
                # Plot activity with bar style for retro feel
                ax.bar(dates, st.session_state.user_data['activity_minutes'], color='white', width=0.6)
                
                # Set y-axis limits
                ax.set_ylim(0, max(120, st.session_state.user_data['activity_minutes'].max() + 10))
                
                # Add grid for retro feel
                ax.grid(True, linestyle='--', alpha=0.7, axis='y')
                
                # Set labels
                ax.set_ylabel('Minutes')
                ax.set_title('Activity History')
                
                # Remove spines
                for spine in ['top', 'right']:
                    ax.spines[spine].set_visible(False)
                
                st.pyplot(fig)
            else:
                st.markdown('<p>No activity data available</p>', unsafe_allow_html=True)
        
        with tab4:
            if 'symptoms' in st.session_state.user_data.columns:
                symptom_data = st.session_state.user_data[['date', 'symptoms']].copy()
                symptom_data = symptom_data[symptom_data['symptoms'].notna() & (symptom_data['symptoms'] != '')]
                
                if len(symptom_data) > 0:
                    st.markdown('<h3>REPORTED SYMPTOMS</h3>', unsafe_allow_html=True)
                    
                    # Display symptoms with retro styling
                    for _, row in symptom_data.iterrows():
                        st.markdown(f"""
                        <div style="border: 2px solid white; padding: 0.5rem; margin-bottom: 0.5rem;">
                            <p style="margin: 0;"><strong>{row['date']}:</strong> {row['symptoms']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.markdown('<p>No symptoms reported</p>', unsafe_allow_html=True)
            else:
                st.markdown('<p>No symptom data available</p>', unsafe_allow_html=True)
        
        # Quick insights
        st.markdown('<h2>QUICK INSIGHTS</h2>', unsafe_allow_html=True)
        
        insights = generate_trend_insight(st.session_state.user_data)
        if isinstance(insights, list):
            for insight in insights:
                st.markdown(f"""
                <div style="border: 2px solid white; padding: 1rem; margin-bottom: 0.5rem;">
                    <p style="margin: 0;">» {insight}</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="border: 2px solid white; padding: 1rem;">
                <p style="margin: 0;">» {insights}</p>
            </div>
            """, unsafe_allow_html=True)

# Add data page
elif st.session_state.page == 'add_data':
    st.markdown('<h1 class="scanlines">ADD HEALTH DATA</h1>', unsafe_allow_html=True)
    
    with st.form("health_data_form"):
        st.markdown('<h3>TODAY\'S STATS</h3>', unsafe_allow_html=True)
        
        # Form inputs
        col1, col2 = st.columns(2)
        
        with col1:
            date = st.date_input("Date", datetime.date.today())
            mood = st.slider("Mood (0-10)", 0, 10, 5)
            stress = st.slider("Stress Level (0-10)", 0, 10, 5)
        
        with col2:
            sleep_hours = st.slider("Sleep Hours", 0, 12, 7)
            activity_minutes = st.slider("Activity Minutes", 0, 180, 30)
            symptoms = st.text_area("Symptoms (if any)")
        
        # Submit button
        submit = st.form_submit_button("SAVE DATA")
        
        if submit:
            # Create entry
            entry = {
                'date': date.strftime('%Y-%m-%d'),
                'mood': mood,
                'stress': stress,
                'sleep_hours': sleep_hours,
                'activity_minutes': activity_minutes,
                'symptoms': symptoms
            }
            
            # Add to dataframe
            st.session_state.user_data = add_entry(st.session_state.user_data, entry)
            
            # Save data
            save_data(st.session_state.user_data, st.session_state.current_user)
            
            st.success("Health data saved successfully!")

# Insights page
elif st.session_state.page == 'insights':
    st.markdown('<h1 class="scanlines">HEALTH INSIGHTS</h1>', unsafe_allow_html=True)
    
    if len(st.session_state.user_data) < 5:
        st.markdown("""
        <div style="text-align: center; padding: 2rem;">
            <h2>Not enough data</h2>
            <p>Add at least 5 entries to generate insights</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Trends
        st.markdown('<h2>DETECTED TRENDS</h2>', unsafe_allow_html=True)
        
        insights = generate_trend_insight(st.session_state.user_data)
        if isinstance(insights, list):
            for insight in insights:
                st.markdown(f"""
                <div style="border: 2px solid white; padding: 1rem; margin-bottom: 0.5rem;">
                    <p style="margin: 0; font-size: 1.2rem;">» {insight}</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="border: 2px solid white; padding: 1rem;">
                <p style="margin: 0; font-size: 1.2rem;">» {insights}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Anomalies
        st.markdown('<h2>ANOMALY DETECTION</h2>', unsafe_allow_html=True)
        
        anomalies = detect_anomalies(st.session_state.user_data)
        if isinstance(anomalies, list):
            for anomaly in anomalies:
                st.markdown(f"""
                <div style="border: 2px solid white; padding: 1rem; margin-bottom: 0.5rem; background-color: #1a1a1a;">
                    <p style="margin: 0; font-size: 1.2rem;">! {anomaly}</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="border: 2px solid white; padding: 1rem;">
                <p style="margin: 0; font-size: 1.2rem;">! {anomalies}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Basic correlation analysis
        st.markdown('<h2>CORRELATION ANALYSIS</h2>', unsafe_allow_html=True)
        
        if 'mood' in st.session_state.user_data.columns and 'sleep_hours' in st.session_state.user_data.columns:
            # Create retro-style scatter plot
            fig, ax = plt.subplots(figsize=(8, 8))
            fig, ax = configure_plot_for_dark_theme(fig, ax)
            
            # Plot scatter with square markers for retro feel
            ax.scatter(st.session_state.user_data['sleep_hours'], st.session_state.user_data['mood'], 
                      s=100, marker='s', color='white')
            
            # Add grid for retro feel
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # Set labels
            ax.set_xlabel('Sleep Hours')
            ax.set_ylabel('Mood Level')
            ax.set_title('Sleep vs. Mood')
            
            # Set axis limits with a bit of padding
            ax.set_xlim(0, max(12, st.session_state.user_data['sleep_hours'].max() + 1))
            ax.set_ylim(0, 11)
            
            # Remove spines
            for spine in ['top', 'right']:
                ax.spines[spine].set_visible(False)
            
            # Add correlation line if there are enough points
            if len(st.session_state.user_data) >= 3:
                # Prepare data for linear regression
                X = st.session_state.user_data['sleep_hours'].values.reshape(-1, 1)
                y = st.session_state.user_data['mood'].values
                
                # Fit linear regression model
                model = LinearRegression()
                model.fit(X, y)
                
                # Generate points for the line
                x_range = np.linspace(0, max(12, st.session_state.user_data['sleep_hours'].max()), 100).reshape(-1, 1)
                y_pred = model.predict(x_range)
                
                # Plot line
                ax.plot(x_range, y_pred, 'w--', linewidth=2)
                
                # Add correlation coefficient
                corr = st.session_state.user_data['sleep_hours'].corr(st.session_state.user_data['mood'])
                ax.text(0.05, 0.95, f"Correlation: {corr:.2f}", transform=ax.transAxes,
                        fontsize=12, verticalalignment='top')
            
            st.pyplot(fig)
        
        if 'stress' in st.session_state.user_data.columns and 'activity_minutes' in st.session_state.user_data.columns:
            # Create retro-style scatter plot
            fig, ax = plt.subplots(figsize=(8, 8))
            fig, ax = configure_plot_for_dark_theme(fig, ax)
            
            # Plot scatter with square markers for retro feel
            ax.scatter(st.session_state.user_data['activity_minutes'], st.session_state.user_data['stress'], 
                      s=100, marker='s', color='white')
            
            # Add grid for retro feel
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # Set labels
            ax.set_xlabel('Activity Minutes')
            ax.set_ylabel('Stress Level')
            ax.set_title('Activity vs. Stress')
            
            # Set axis limits with a bit of padding
            ax.set_xlim(0, max(120, st.session_state.user_data['activity_minutes'].max() + 10))
            ax.set_ylim(0, 11)
            
            # Remove spines
            for spine in ['top', 'right']:
                ax.spines[spine].set_visible(False)
            
            # Add correlation line if there are enough points
            if len(st.session_state.user_data) >= 3:
                # Prepare data for linear regression
                X = st.session_state.user_data['activity_minutes'].values.reshape(-1, 1)
                y = st.session_state.user_data['stress'].values
                
                # Fit linear regression model
                model = LinearRegression()
                model.fit(X, y)
                
                # Generate points for the line
                x_range = np.linspace(0, max(120, st.session_state.user_data['activity_minutes'].max()), 100).reshape(-1, 1)
                y_pred = model.predict(x_range)
                
                # Plot line
                ax.plot(x_range, y_pred, 'w--', linewidth=2)
                
                # Add correlation coefficient
                corr = st.session_state.user_data['activity_minutes'].corr(st.session_state.user_data['stress'])
                ax.text(0.05, 0.95, f"Correlation: {corr:.2f}", transform=ax.transAxes,
                        fontsize=12, verticalalignment='top')
            
            st.pyplot(fig)

# Settings page
elif st.session_state.page == 'settings':
    st.markdown('<h1 class="scanlines">SETTINGS</h1>', unsafe_allow_html=True)
    
    # Export data
    st.markdown('<h2>EXPORT YOUR DATA</h2>', unsafe_allow_html=True)
    
    if len(st.session_state.user_data) > 0:
        csv = st.session_state.user_data.to_csv(index=False)
        st.download_button(
            label="DOWNLOAD CSV",
            data=csv,
            file_name=f"{st.session_state.current_user}_health_data.csv",
            mime="text/csv",
        )
    else:
        st.markdown('<p>No data to export</p>', unsafe_allow_html=True)
    
    # Clear data option
    st.markdown('<h2>DANGER ZONE</h2>', unsafe_allow_html=True)
    
    if st.button("CLEAR ALL DATA"):
        confirm = st.checkbox("I understand this will delete all my health data")
        
        if confirm:
            if st.button("CONFIRM DELETE"):
                st.session_state.user_data = pd.DataFrame(columns=['date', 'mood', 'stress', 'sleep_hours', 'activity_minutes', 'symptoms'])
                save_data(st.session_state.user_data, st.session_state.current_user)
                st.success("All data cleared successfully!")

# Footer
st.markdown("""
<div style="text-align: center; margin-top: 3rem; padding-top: 1rem; border-top: 2px solid white;">
    <p>PROJECT SYNAPSE CORE v0.1.0 | YOUR PERSONAL HEALTH TRACKER</p>
</div>
""", unsafe_allow_html=True)