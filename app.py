# app.py (v3 - Crash Fixed and Warnings Fixed)
import streamlit as st
import pandas as pd
import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# --- Page Configuration ---
st.set_page_config(
    page_title="AI Job Impact Analysis",
    page_icon="🤖",
    layout="wide",
)

# --- Data Loading ---
@st.cache_data
def load_data():
    try:
        econ_data = pd.read_csv('final_econ_data.csv')
    except FileNotFoundError:
        st.error("Error: 'final_econ_data.csv' not found. Make sure it's in the same directory as app.py")
        return None, None, None
    
    try:
        job_profile_long = pd.read_csv('final_job_profile_long.csv')
    except FileNotFoundError:
        st.error("Error: 'final_job_profile_long.csv' not found.")
        return None, None, None
        
    try:
        skill_signature = pd.read_csv('final_skill_signature.csv').set_index('KMeans_Cluster')
    except FileNotFoundError:
        st.error("Error: 'final_skill_signature.csv' not found.")
        return None, None, None
    
    if econ_data is not None:
        top_social_threshold = econ_data['Social_Skills'].quantile(0.75)
        econ_data['Insulation (Social Skills)'] = np.where(
            econ_data['Social_Skills'] >= top_social_threshold, 
            'High (Insulated)', 
            'Low'
        )

    return econ_data, job_profile_long, skill_signature

df_final_econ, job_profile_long_df, scaled_skill_profile = load_data()

if df_final_econ is None:
    st.stop()


# --- Main App ---
st.title("🤖 Analysis of AI's Impact on the Job Market")
st.markdown("""
This analysis explores the potential impact of AI on different occupations based on their core skill requirements. 
We've clustered jobs into archetypes and analyzed their vulnerability to generative AI and robotics. 
All plots are interactive: **hover for details, click and drag to pan, scroll to zoom.**
""")

# --- Create Tabs for Each Section ---
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "1. AI Impact Synthesis", 
    "2. AI Risk by Job Cluster", 
    "3. Economic Risk Quadrants", 
    "4. Cluster Skill DNA", 
    "5. Job Skill Explorer"
])


# --- Tab 1: AI Impact Synthesis Plot ---
with tab1:
    st.header("1. Interactive AI Impact Synthesis Plot")
    st.markdown("""
    This plot maps the job market based on two key risk factors:
    * **X-axis (Routine Intensity):** How repetitive or procedural a job is. **Higher is more risk.**
    * **Y-axis (Mental Prowess):** How much complex analysis is required. **Higher is more GenAI risk.**
    
    The color shows insulation:
    * **Green (High Social Skills):** Jobs requiring negotiation, empathy, and coordination are **more insulated**.
    * **Red (Low Social Skills):** Jobs that are more technical or analytical are **less insulated**.
    * **Bubble Size:** Represents the job's **Physical Prowess**.
    """)

    base = alt.Chart(df_final_econ).encode(
        x=alt.X('Routine_Intensity', title='Routine Task Intensity (Higher = More Risk)'),
        y=alt.Y('Mental_Prowess', title='Mental Prowess (Higher = GenAI Risk)'),
        tooltip=[
            'Title', 'Mental_Prowess', 'Physical_Prowess',
            'Social_Skills', 'Routine_Intensity', 'KMeans_Cluster'
        ],
        color=alt.Color('Insulation (Social Skills)',
                        scale={'domain': ['High (Insulated)', 'Low'],
                               'range': ['green', 'darkred']}),
        size=alt.Size('Physical_Prowess', title='Physical Prowess', legend=None)
    ).properties(
        title='Interactive AI Impact Synthesis Plot'
    ).interactive()

    mean_routine = df_final_econ['Routine_Intensity'].mean()
    mean_mental = df_final_econ['Mental_Prowess'].mean()
    vertical_line = alt.Chart(pd.DataFrame({'x': [mean_routine]})).mark_rule(strokeDash=[2, 2], color='gray').encode(x='x')
    horizontal_line = alt.Chart(pd.DataFrame({'y': [mean_mental]})).mark_rule(strokeDash=[2, 2], color='gray').encode(y='y')

    synthesis_chart = base.mark_circle(opacity=0.6) + horizontal_line + vertical_line

    # --- FIX 2: Added width='stretch' ---
    st.altair_chart(synthesis_chart, width='stretch')

# --- Tab 2: Breakdown of AI Impact Risk by Job Cluster ---
with tab2:
    st.header("2. Breakdown of AI Impact Risk by Job Cluster")
    st.markdown("""
    This chart shows the *composition* of each job cluster, broken down by our defined AI risk timelines.
    It answers the question: "What kind of AI risk (e.g., Near-Term Generative AI vs. Medium-Term Robotics) 
    is most common in each job sector?"
    """)

    impact_breakdown = df_final_econ.groupby(['KMeans_Cluster', 'AI_Impact_Type']).size().reset_index(name='Job Count')
    
    breakdown_chart = alt.Chart(impact_breakdown).mark_bar().encode(
        x=alt.X('KMeans_Cluster:N', title='Job Cluster (K-Means)'),
        y=alt.Y('Job Count', stack='normalize', title='Percentage of Jobs', axis=alt.Axis(format='%')),
        color=alt.Color('AI_Impact_Type', title='AI Risk Type (Timeline)'),
        tooltip=['KMeans_Cluster', 'AI_Impact_Type', 'Job Count']
    ).properties(
        title='Breakdown of AI Impact Risk by Job Cluster'
    ).interactive()
    
    # --- FIX 2: Added width='stretch' ---
    st.altair_chart(breakdown_chart, width='stretch')

# --- Tab 3: Economic Risk Quadrant ---
with tab3:
    st.header("3. Economic Analysis of AI Job Risk")
    st.markdown("""
    This chart plots job risk against its economic value (wage) and scale (total employment).
    
    * **X-axis (AI Impact Score):** Our calculated risk score (Routine - Social). **Higher is more risk.**
    * **Y-axis (Median Annual Wage):** The average salary.
    * **Bubble Size:** Represents **Total Employment** (in millions).
    """)

    df_plot_data = df_final_econ.copy()
    df_plot_data['Total_Employment_Millions'] = df_plot_data['Total_Employment_USD'] / 1_000_000

    econ_chart_base = alt.Chart(df_plot_data).mark_circle(opacity=0.7).encode(
        x=alt.X('AI_Impact_Score', scale=alt.Scale(zero=False), title='AI Impact Score (Higher = More Risk)'),
        y=alt.Y('Median_Annual_Wage_USD', scale=alt.Scale(zero=False), title='Median Annual Wage (USD)', axis=alt.Axis(format='$,.0f')),
        color=alt.Color('AI_Impact_Type', title='AI Risk Profile'),
        size=alt.Size('Total_Employment_Millions', title='Total Employment (Millions)', legend=alt.Legend(format='.1f')),
        tooltip=[
            'Title', 'AI_Impact_Type',
            alt.Tooltip('AI_Impact_Score', format='.2f'),
            alt.Tooltip('Median_Annual_Wage_USD', format='$,.0f'),
            alt.Tooltip('Total_Employment_Millions', format='.2f')
        ]
    ).properties(
        title='Economic Analysis of AI Job Risk'
    ).interactive()

    mean_risk = df_plot_data['AI_Impact_Score'].mean()
    mean_wage = df_plot_data['Median_Annual_Wage_USD'].mean()
    v_line_econ = alt.Chart(pd.DataFrame({'x': [mean_risk]})).mark_rule(strokeDash=[2, 2], color='gray').encode(x='x')
    h_line_econ = alt.Chart(pd.DataFrame({'y': [mean_wage]})).mark_rule(strokeDash=[2, 2], color='gray').encode(y='y')

    final_econ_chart = econ_chart_base + v_line_econ + h_line_econ
    
    # --- FIX 2: Added width='stretch' ---
    st.altair_chart(final_econ_chart, width='stretch')

    st.markdown("### Takeaway: Economic Risk Quadrant Analysis")
    st.markdown("""
    This chart divides the job market into four quadrants:
    * **Top-Right (High Wage, High Risk):** "White-Collar" exposure. High-paid, routine jobs (e.g., Financial Analysts, Programmers) that companies are incentivized to augment with Generative AI.
    * **Bottom-Right (Low Wage, High Risk):** "Robotics & Automation" risk. Low-paid, routine, physical jobs (e.g., Assemblers, Data Entry). This often represents a large number of workers.
    * **Top-Left (High Wage, Low Risk):** "Insulated" professionals. High-paid, creative, and high-social-skill jobs (e.g., Chief Executives, Surgeons) where AI is a tool, not a replacement.
    * **Bottom-Left (Low Wage, Low Risk):** "Service" insulation. Low-paid but high-touch, non-routine service jobs (e.g., Childcare Workers, Hairstylists) that are difficult to automate.
    """)

# --- Tab 4: Skill Signature Heatmap ---
with tab4:
    st.header("4. Cluster Skill Signature (Cluster DNA)")
    st.markdown("""
    This heatmap (which is not interactive) shows the unique "skill signature" of each job cluster.
    It answers: "What skills *define* this group?"
    * **Bright Red (High Positive Score):** A defining, unique skill for that cluster.
    * **Bright Blue (High Negative Score):** A skill this cluster is uniquely lacking.
    * **White (Near Zero):** An "average" skill for this cluster, not a defining feature.
    """)

    if 'scaled_skill_profile' in locals():
        def plot_skill_heatmap():
            fig, ax = plt.subplots(figsize=(20, 10))
            sns.heatmap(
                scaled_skill_profile,
                annot=True, cmap='vlag', center=0,
                fmt='.2f', linewidths=.5, ax=ax
            )
            ax.set_title('Skill Signature Heatmap by K-Means Cluster (Z-Scores)', fontsize=20, pad=20)
            ax.set_xlabel('Individual Skills', fontsize=14)
            ax.set_ylabel('K-Means Cluster', fontsize=14)
            plt.xticks(rotation=45, ha='right')
            return fig

        # --- FIX 2: Added use_container_width=True (st.pyplot is different) ---
        st.pyplot(plot_skill_heatmap(), use_container_width=True)

# --- Tab 5: Interactive Job Explorer ---
with tab5:
    st.header("5. Interactive Job Skill Explorer")
    st.markdown("Select any job title from the dropdown menu to see its detailed skill profile and risk assessment.")

    if 'job_profile_long_df' in locals():
        
        job_titles = sorted(job_profile_long_df['Title'].unique())
        
        selected_job = st.selectbox(
            'Select a Job Title:',
            options=job_titles,
            # --- FIX 1: This is the fix for the crash ---
            index=0 # Default to the first job in the list
        )

        job_data = job_profile_long_df[job_profile_long_df['Title'] == selected_job]

        if not job_data.empty:
            details = job_data.iloc[0]
            st.subheader(f"Profile for: {details['Title']}")
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Job Cluster", f"Cluster {details['KMeans_Cluster']}")
            col2.metric("Median Annual Wage", f"${details['Median_Annual_Wage_USD']:,.0f}")
            col3.metric("AI Risk Profile", details['AI_Impact_Type'])

            skill_chart = alt.Chart(job_data).mark_bar().encode(
                x=alt.X('Skill_Value:Q', title='Skill Level (O*NET Score)'),
                y=alt.Y('Skill:N', sort='-x'),
                tooltip=['Skill', 'Skill_Value']
            ).properties(
                title=f'Skill Breakdown for {selected_job}'
            ).interactive()
            
            # --- FIX 2: Added width='stretch' ---
            st.altair_chart(skill_chart, width='stretch')