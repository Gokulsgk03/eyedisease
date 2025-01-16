import streamlit as st
import PyPDF2
from io import BytesIO
import pandas as pd
import plotly.express as px
from datetime import datetime
import re
import plotly.graph_objects as go
import numpy as np
from plotly.subplots import make_subplots
import streamlit as st

class EnhancedVisualizations:
    def __init__(self):
        self.eye_colors = {
            'sclera': '#ffffff',
            'iris': '#4287f5',
            'pupil': '#000000',
            'cornea': '#e6f3ff',
            'highlight': '#ffffff'
        }

    def create_3d_eye_model(self):
        """Create an interactive 3D eye model"""
        # Create a sphere for the eyeball
        phi = np.linspace(0, 2*np.pi, 100)
        theta = np.linspace(-np.pi/2, np.pi/2, 100)
        phi, theta = np.meshgrid(phi, theta)

        r = 1
        x = r * np.cos(theta) * np.cos(phi)
        y = r * np.cos(theta) * np.sin(phi)
        z = r * np.sin(theta)

        # Create the 3D eye model
        fig = go.Figure()

        # Add the sclera (white part)
        fig.add_trace(go.Surface(
            x=x, y=y, z=z,
            colorscale=[[0, self.eye_colors['sclera']], [1, self.eye_colors['sclera']]],
            showscale=False,
            name='Sclera'
        ))

        # Add the iris
        iris_r = 0.5
        iris_x = iris_r * np.cos(theta) * np.cos(phi)
        iris_y = iris_r * np.cos(theta) * np.sin(phi)
        iris_z = np.ones_like(iris_x)

        fig.add_trace(go.Surface(
            x=iris_x, y=iris_y, z=iris_z,
            colorscale=[[0, self.eye_colors['iris']], [1, self.eye_colors['iris']]],
            showscale=False,
            name='Iris'
        ))

        # Update layout
        fig.update_layout(
            title='Interactive 3D Eye Model',
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
                camera=dict(
                    up=dict(x=0, y=0, z=1),
                    center=dict(x=0, y=0, z=0),
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            width=600,
            height=600
        )

        return fig

    def create_interactive_anatomy(self):
        """Create interactive anatomical diagrams"""
        # Create a side view of the eye with annotations
        fig = go.Figure()

        # Add the main eye outline
        t = np.linspace(0, 2*np.pi, 100)
        x = np.cos(t)
        y = np.sin(t)

        fig.add_trace(go.Scatter(
            x=x, y=y,
            mode='lines',
            line=dict(color='black', width=2),
            name='Eye Outline'
        ))

        # Add annotations for different parts
        annotations = [
            dict(x=1.2, y=0, text='Cornea', showarrow=True, arrowhead=2),
            dict(x=0, y=1.2, text='Sclera', showarrow=True, arrowhead=2),
            dict(x=0.5, y=0, text='Lens', showarrow=True, arrowhead=2),
            dict(x=-0.8, y=0, text='Retina', showarrow=True, arrowhead=2),
            dict(x=0, y=-1.2, text='Optic Nerve', showarrow=True, arrowhead=2)
        ]

        fig.update_layout(
            title='Interactive Eye Anatomy',
            annotations=annotations,
            showlegend=False,
            width=600,
            height=600,
            xaxis=dict(range=[-2, 2], showgrid=False),
            yaxis=dict(range=[-2, 2], showgrid=False)
        )

        return fig

    def create_vision_field_heatmap(self, data=None):
        """Create a heatmap of vision field test results"""
        if data is None:
            # Generate sample data if none provided
            x = np.linspace(-30, 30, 20)
            y = np.linspace(-30, 30, 20)
            X, Y = np.meshgrid(x, y)
            Z = np.exp(-(X**2 + Y**2)/400)  # Gaussian distribution for sample data

        fig = go.Figure(data=go.Heatmap(
            z=Z,
            colorscale='Viridis',
            showscale=True
        ))

        fig.update_layout(
            title='Vision Field Test Heatmap',
            xaxis_title='Horizontal Field (degrees)',
            yaxis_title='Vertical Field (degrees)',
            width=600,
            height=600
        )

        return fig

    def create_visit_comparison(self, visits_data):
        """Create side-by-side comparison of visits"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Vision Scores', 'Pressure Readings', 
                          'Condition Status', 'Risk Assessment')
        )

        # Vision Scores
        fig.add_trace(
            go.Scatter(
                x=[visit['date'] for visit in visits_data],
                y=[visit.get('measurements', {}).get('vision_score', 0) for visit in visits_data],
                name='Vision Score',
                mode='lines+markers'
            ),
            row=1, col=1
        )

        # Pressure Readings
        fig.add_trace(
            go.Scatter(
                x=[visit['date'] for visit in visits_data],
                y=[visit.get('measurements', {}).get('pressure_right', 0) for visit in visits_data],
                name='Right Eye Pressure',
                mode='lines+markers'
            ),
            row=1, col=2
        )
        fig.add_trace(
            go.Scatter(
                x=[visit['date'] for visit in visits_data],
                y=[visit.get('measurements', {}).get('pressure_left', 0) for visit in visits_data],
                name='Left Eye Pressure',
                mode='lines+markers'
            ),
            row=1, col=2
        )

        # Update layout
        fig.update_layout(
            height=800,
            width=1000,
            title_text="Visit Comparison Dashboard",
            showlegend=True
        )

        return fig

# Add this to your main application:
def add_visualization_section():
    st.header("Advanced Visualizations")
    
    viz = EnhancedVisualizations()
    
    # Create tabs for different visualizations
    viz_tabs = st.tabs(["3D Model", "Anatomy", "Vision Field", "Visit Comparison"])
    
    with viz_tabs[0]:
        st.plotly_chart(viz.create_3d_eye_model())
        st.markdown("""
            This 3D model shows the basic structure of the human eye. 
            You can rotate and zoom using your mouse/touchpad.
        """)

    with viz_tabs[1]:
        st.plotly_chart(viz.create_interactive_anatomy())
        st.markdown("""
            This diagram shows the key anatomical features of the eye.
            Hover over different parts to learn more about them.
        """)

    with viz_tabs[2]:
        st.plotly_chart(viz.create_vision_field_heatmap())
        st.markdown("""
            This heatmap represents the vision field test results.
            Brighter colors indicate better vision in that area.
        """)

    with viz_tabs[3]:
        if st.session_state.patient_history:
            st.plotly_chart(viz.create_visit_comparison(st.session_state.patient_history))
        else:
            st.warning("No visit history available for comparison.")


# PDF Processing Functions
def extract_pdf_content(pdf_file):
    """Extract text content from PDF with error handling"""
    try:
        pdf_bytes = pdf_file.read()
        pdf_reader = PyPDF2.PdfReader(BytesIO(pdf_bytes))
        content = ""
        with st.spinner('Extracting text from PDF...'):
            for page in pdf_reader.pages:
                content += page.extract_text()
        pdf_file.seek(0)  # Reset file pointer
        return content
    except Exception as e:
        st.error(f"Error processing PDF file: {str(e)}")
        return None

def validate_eye_report(content):
    """Validate if the content is from an eye report"""
    keywords = ['vision', 'eye', 'ophthalmology', 'retina', 'cornea', 'iop', 'pressure']
    return any(keyword in content.lower() for keyword in keywords)

# Data Management Functions
def load_patient_history():
    if 'patient_history' not in st.session_state:
        st.session_state.patient_history = []

def save_report_to_history(report_data):
    st.session_state.patient_history.append(report_data)

def calculate_vision_progression(history):
    """Calculate vision progression from historical data"""
    if not history:
        return [], []
    dates = [entry['date'] for entry in history]
    vision_values = [entry.get('measurements', {}).get('vision_score', 0) for entry in history]
    return dates, vision_values

class EyeReportAnalyzer:
    def __init__(self):
        self.severity_scale = {
            'Mild': 1,
            'Moderate': 2,
            'Severe': 3
        }

    def extract_measurements(self, content):
        """Enhanced measurement extraction with better pattern matching"""
        measurements = {}
        
        # Improved pressure pattern matching
        pressure_pattern = r"(?:Pressure|IOP|Intraocular Pressure)[:\s].*?(?:Right|OD)[:\s]*?(\d{1,2}).*?(?:Left|OS)[:\s]*?(\d{1,2})"
        pressure_match = re.search(pressure_pattern, content, re.IGNORECASE | re.DOTALL)
        if pressure_match:
            measurements['pressure_right'] = int(pressure_match.group(1))
            measurements['pressure_left'] = int(pressure_match.group(2))
        
        # Improved vision pattern matching
        vision_pattern = r"(?:Visual Acuity|Vision)[:\s].*?(?:Right|OD)[:\s]*?(20/\d{1,3}).*?(?:Left|OS)[:\s]*?(20/\d{1,3})"
        vision_match = re.search(vision_pattern, content, re.IGNORECASE | re.DOTALL)
        if vision_match:
            measurements['vision_right'] = vision_match.group(1)
            measurements['vision_left'] = vision_match.group(2)
            # Calculate vision scores
            right_score = 20 / int(vision_match.group(1).split('/')[1])
            left_score = 20 / int(vision_match.group(2).split('/')[1])
            measurements['vision_score'] = round((right_score + left_score) / 2, 2)
            measurements['vision_difference'] = abs(right_score - left_score)
        
        measurements['date'] = datetime.now().strftime("%Y-%m-%d")
        return measurements

    def analyze_conditions(self, content):
        """Enhanced condition analysis with progression tracking"""
        conditions = []
        
        # Define condition patterns with status indicators
        condition_patterns = {
            'myopia': {
                'pattern': r'(?:myopia|nearsighted)',
                'severity_pattern': r'(mild|moderate|severe)\s+(?:myopia|nearsighted)',
                'progression_pattern': r'(stable|progressive|worsening)\s+(?:myopia|nearsighted)'
            },
            'cataract': {
                'pattern': r'(?:cataract|lens opacity)',
                'severity_pattern': r'((?:early|mild|moderate|severe))\s+(?:cataract|lens opacity)',
                'progression_pattern': r'(stable|progressive|worsening)\s+(?:cataract|lens opacity)'
            },
            'dry_eye': {
                'pattern': r'dry\s+eye',
                'severity_pattern': r'(mild|moderate|severe)\s+dry\s+eye',
                'progression_pattern': r'(improving|stable|worsening)\s+dry\s+eye'
            },
            'astigmatism': {
                'pattern': r'astigmatism',
                'severity_pattern': r'(mild|moderate|severe)\s+astigmatism',
                'progression_pattern': r'(stable|progressive|worsening)\s+astigmatism'
            }
        }
        
        for condition, patterns in condition_patterns.items():
            if re.search(patterns['pattern'], content.lower()):
                severity = 'Moderate'  # Default severity
                status = 'Stable'      # Default status
                
                # Check for specific severity mentions
                severity_match = re.search(patterns['severity_pattern'], content.lower())
                if severity_match and severity_match.group(1):
                    severity = severity_match.group(1).capitalize()
                
                # Check for progression status
                progression_match = re.search(patterns['progression_pattern'], content.lower())
                if progression_match and progression_match.group(1):
                    status = progression_match.group(1).capitalize()
                
                conditions.append({
                    'condition': condition.replace('_', ' ').title(),
                    'severity': severity,
                    'status': status
                })
        
        return conditions

    def generate_recommendations(self, conditions, measurements):
        """Enhanced recommendations based on specific measurements and conditions"""
        short_term = []
        long_term = []
        
        # Pressure-based recommendations
        if measurements.get('pressure_right') or measurements.get('pressure_left'):
            right_pressure = measurements.get('pressure_right', 0)
            left_pressure = measurements.get('pressure_left', 0)
            
            if right_pressure > 21 or left_pressure > 21:
                short_term.append(
                    f"URGENT: Your eye pressure readings (Right: {right_pressure} mmHg, Left: {left_pressure} mmHg) "
                    "show elevated levels requiring immediate attention. Schedule a follow-up within 2-4 weeks for "
                    "pressure monitoring and possible glaucoma assessment."
                )
        
        # Vision difference recommendations
        if measurements.get('vision_difference'):
            if measurements['vision_difference'] > 0.2:
                short_term.append(
                    f"The difference in vision between your eyes requires attention. "
                    f"Right eye ({measurements['vision_right']}) and Left eye ({measurements['vision_left']}) "
                    "show significant variance. Consider updating your prescription and discussing "
                    "options for vision correction with your eye care provider."
                )
        
        # Condition-specific recommendations
        for condition in conditions:
            if condition['condition'] == 'Cataract' and condition['status'] == 'Progressive':
                short_term.append(
                    "Your progressive cataract requires monitoring. Schedule follow-up in 3-4 months. "
                    "Consider discussing surgical options if vision significantly affects daily activities. "
                    "Use good lighting and anti-glare solutions for immediate comfort."
                )
                
            if condition['condition'] == 'Dry Eye':
                if condition['severity'] in ['Moderate', 'Severe']:
                    short_term.append(
                        "For dry eye management: Use preservative-free artificial tears 4-6 times daily. "
                        "Consider punctal plugs and maintain humidity levels in your environment. "
                        "Take regular breaks during screen use and practice the 20-20-20 rule."
                    )
        
        # Long-term health recommendations
        long_term.append(
            "Develop a comprehensive eye health plan including:\n"
            "1. Regular eye pressure checks every 3-4 months\n"
            "2. Annual comprehensive eye exams\n"
            "3. Diet rich in eye-healthy nutrients (leafy greens, omega-3 fatty acids)\n"
            "4. Proper eye protection from UV and blue light\n"
            "5. Regular exercise and maintaining healthy blood pressure"
        )
        
        return short_term, long_term

    def generate_summary(self, measurements, conditions, short_term_recs, long_term_recs):
        """Generate enhanced summary with all key findings"""
        summary = []
        
        # Detailed examination summary
        summary.append("EXAMINATION SUMMARY:")
        
        if measurements:
            if 'vision_right' in measurements and 'vision_left' in measurements:
                summary.append(
                    f"\nVISION STATUS:\n"
                    f"- Right Eye: {measurements['vision_right']}\n"
                    f"- Left Eye: {measurements['vision_left']}\n"
                    f"- Overall Vision Score: {measurements.get('vision_score', 'N/A')}"
                )
            
            if 'pressure_right' in measurements and 'pressure_left' in measurements:
                summary.append(
                    f"\nPRESSURE READINGS:\n"
                    f"- Right Eye: {measurements['pressure_right']} mmHg\n"
                    f"- Left Eye: {measurements['pressure_left']} mmHg"
                )
        
        if conditions:
            summary.append("\nDIAGNOSED CONDITIONS:")
            for condition in conditions:
                summary.append(
                    f"- {condition['condition']}: {condition['severity']} ({condition['status']})"
                )
        
        summary.append("\nKEY ACTION ITEMS:")
        if short_term_recs:
            summary.append("\nIMMEDIATE ACTIONS REQUIRED:")
            for rec in short_term_recs:
                summary.append(f"- {rec}")
        
        if long_term_recs:
            summary.append("\nLONG-TERM MANAGEMENT:")
            for rec in long_term_recs:
                summary.append(f"- {rec}")
        
        return "\n".join(summary)

def create_vision_chart(dates, values):
    """Create vision progression chart"""
    fig = px.line(
        x=dates,
        y=values,
        title='Vision Score Progression',
        labels={'x': 'Date', 'y': 'Vision Score'},
        template='plotly_white'
    )
    fig.update_traces(line_color='#2E86C1', line_width=2)
    fig.update_layout(
        hovermode='x unified',
        plot_bgcolor='white',
        showlegend=False
    )
    return fig

def main():
    st.set_page_config(page_title="Advanced Eye Report Analyzer", layout="wide")
    
    # Initialize
    load_patient_history()
    analyzer = EyeReportAnalyzer()
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Report Analysis", "History & Trends", "Educational Resources"])
    
    if page == "Report Analysis":
        st.title("👁️ Advanced Eye Report Analysis")
        
        uploaded_file = st.file_uploader("Upload Eye Report", type=['pdf'])
        if uploaded_file:
            content = extract_pdf_content(uploaded_file)
            
            if content:
                if not validate_eye_report(content):
                    st.error("This doesn't appear to be an eye report. Please check your document.")
                    st.stop()
                
                report_date = st.date_input("Report Date", datetime.now())
                symptoms = st.multiselect(
                    "Select any current symptoms",
                    ["Blurred Vision", "Eye Pain", "Redness", "Light Sensitivity", "Floaters"]
                )
                
                if st.button("Analyze Report"):
                    with st.spinner("Analyzing report..."):
                        measurements = analyzer.extract_measurements(content)
                        conditions = analyzer.analyze_conditions(content)
                        short_term_recs, long_term_recs = analyzer.generate_recommendations(conditions, measurements)
                        
                        st.subheader("📋 Executive Summary")
                    
                    summary = analyzer.generate_summary(measurements, conditions, short_term_recs, long_term_recs)
                    st.write(summary)
                    
                    # Save to history
                    if measurements:
                        save_report_to_history({
                            'date': report_date.strftime("%Y-%m-%d"),
                            'measurements': measurements,
                            'conditions': conditions
                        })
                        
                        # Add visualizations after the analysis section
                        if st.checkbox("Show Advanced Visualizations"):
                            add_visualization_section()
                        
                        # Create columns for better organization
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader("📊 Key Measurements")
                            if measurements:
                                for key, value in measurements.items():
                                    if key != 'date':
                                        st.metric(label=key.replace('_', ' ').title(), value=value)
                        
                        with col2:
                            st.subheader("🔍 Identified Conditions")
                            if conditions:
                                st.dataframe(pd.DataFrame(conditions))
                    # Recommendations with enhanced styling
                    st.subheader("📝 Detailed Recommendations")
                    
                    for i, rec in enumerate(short_term_recs):
                        with st.expander(f"Short-term Recommendation {i+1}", expanded=True):
                            st.success(rec)
                    
                    for i, rec in enumerate(long_term_recs):
                        with st.expander(f"Long-term Recommendation {i+1}", expanded=True):
                            st.info(rec)
                            
                    # Add severity indicators
                    st.subheader("⚠️ Risk Assessment")
                    if measurements:
                        risk_levels = {
                            'Vision Risk': 'High' if any(int(v.split('/')[1]) > 40 for v in [measurements.get('vision_right', '20/20'), measurements.get('vision_left', '20/20')]) else 'Low',
                            'Pressure Risk': 'High' if any(p > 21 for p in [measurements.get('pressure_right', 0), measurements.get('pressure_left', 0)]) else 'Low',
                            'Overall Health Risk': 'High' if len(conditions) > 2 else 'Low'
                        }
                        
                        for risk, level in risk_levels.items():
                            color = 'red' if level == 'High' else 'green'
                            st.markdown(f"<span style='color:{color}'>{risk}: {level}</span>", unsafe_allow_html=True)
                        
                    # Add symptoms analysis if any were selected
                    if symptoms:
                        st.subheader("🔔 Symptoms Analysis")
                        symptom_severity = {
                            'Blurred Vision': 'High',
                            'Eye Pain': 'High',
                            'Redness': 'Moderate',
                            'Light Sensitivity': 'Moderate',
                            'Floaters': 'Moderate'
                        }
                        
                        for symptom in symptoms:
                            severity = symptom_severity.get(symptom, 'Low')
                            st.warning(f"{symptom}: {severity} Priority - Reported and noted in patient history")

    elif page == "History & Trends":
        st.title("📈 History & Trends Analysis")
    
    if st.session_state.patient_history:
        # Vision Progression Chart
        dates, values = calculate_vision_progression(st.session_state.patient_history)
        if dates and values:
            st.subheader("Vision Score Progression")
            fig = create_vision_chart(dates, values)
            st.plotly_chart(fig, use_container_width=True)
            
        # Historical Data Table
        st.subheader("📋 Visit History")
        history_df = pd.DataFrame(st.session_state.patient_history)
        
        # Format the dataframe for better display
        if not history_df.empty:
            # Create a copy to avoid modifying original data
            processed_df = history_df.copy()
            
            # Process measurements column
            if 'measurements' in processed_df.columns:
                # Extract measurements data and rename columns to avoid conflicts
                measurements_df = pd.json_normalize(processed_df['measurements']).add_prefix('measurement_')
                # Drop the original measurements column
                processed_df = processed_df.drop('measurements', axis=1)
                # Merge with measurements data
                processed_df = pd.concat([processed_df, measurements_df], axis=1)
            
            # Process conditions column
            if 'conditions' in processed_df.columns:
                processed_df['conditions'] = processed_df['conditions'].apply(
                    lambda x: ', '.join([c['condition'] for c in x]) if x else ''
                )
            
            # Remove any remaining duplicate columns
            processed_df = processed_df.loc[:, ~processed_df.columns.duplicated()]
            
            # Reorder columns for better presentation
            important_columns = ['date', 'conditions']
            other_columns = [col for col in processed_df.columns if col not in important_columns]
            final_columns = important_columns + other_columns
            processed_df = processed_df[final_columns]
            
            # Display the processed DataFrame
            st.dataframe(processed_df)
            
            # Statistical Analysis
            st.subheader("📊 Statistical Analysis")
            if len(values) > 1:
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Average Vision Score", f"{sum(values)/len(values):.2f}")
                    st.metric("Visits Recorded", len(values))
                with col2:
                    vision_trend = "Improving" if values[-1] > values[0] else "Declining"
                    st.metric("Vision Trend", vision_trend)
                    st.metric("Score Change", f"{values[-1] - values[0]:.2f}")

                     # Add visit comparison visualization
    if st.session_state.patient_history:
        viz = EnhancedVisualizations()
        st.plotly_chart(viz.create_visit_comparison(st.session_state.patient_history))

    else:  # Educational Resources
        st.title("📚 Educational Resources")
        
        st.subheader("Common Eye Conditions")
        conditions_info = {
            "Myopia (Nearsightedness)": """
                Myopia is a condition where close objects appear clear, but distant objects appear blurry. 
                It occurs when the eye grows too long or the cornea is too curved. Treatment options include:
                - Corrective lenses (glasses or contacts)
                - LASIK surgery
                - Orthokeratology
                """,
            "Glaucoma": """
                Glaucoma is a group of eye conditions that damage the optic nerve. Early detection and treatment
                are crucial to prevent vision loss. Risk factors include:
                - High intraocular pressure
                - Family history
                - Age over 60
                - Certain medical conditions
                """,
            "Cataracts": """
                Cataracts occur when the natural lens of the eye becomes cloudy. Symptoms develop gradually and
                may include:
                - Blurred vision
                - Difficulty with night vision
                - Sensitivity to light
                - Seeing halos around lights
                """
        }
        
        for condition, info in conditions_info.items():
            with st.expander(condition):
                st.markdown(info)
        
        st.subheader("Digital Eye Strain Prevention")
        st.markdown("""
            ### The 20-20-20 Rule
            Every 20 minutes:
            - Take a 20-second break
            - Look at something 20 feet away
            - Blink 20 times
            
            ### Workspace Ergonomics
            - Position your screen at arm's length
            - Screen should be slightly below eye level
            - Use proper lighting to reduce glare
            - Adjust screen brightness to match surroundings
            
            ### Healthy Habits
            - Regular eye exams
            - Proper hydration
            - Balanced diet rich in eye-healthy nutrients
            - Regular exercise
            - Adequate sleep
        """)
        
        st.subheader("🔍 Additional Resources")
        st.markdown("""
            - [American Academy of Ophthalmology](https://www.aao.org)
            - [National Eye Institute](https://www.nei.nih.gov)
            - [Prevent Blindness](https://preventblindness.org)
        """)

if __name__ == "__main__":
    main()