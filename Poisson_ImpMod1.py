import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import io

# Set page configuration
st.set_page_config(
    page_title="Poisson Impedance Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

class PoissonImpedanceAnalyzer:
    def __init__(self, data):
        """
        Initialize the analyzer with well log data
        """
        self.data = data.copy()
        self.depth_column = None
        
        # Find depth column
        depth_candidates = ['DEPTH', 'Depth', 'depth', 'DEPT', 'TD', 'Z']
        for candidate in depth_candidates:
            if candidate in self.data.columns:
                self.depth_column = candidate
                break
        
        # Check if required columns exist
        required_columns = ['Vp', 'Vs', 'Rho']
        missing_columns = [col for col in required_columns if col not in self.data.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Calculate basic impedances
        self.data['Ip'] = self.data['Vp'] * self.data['Rho']  # P-impedance
        self.data['Is'] = self.data['Vs'] * self.data['Rho']  # S-impedance
    
    def get_depth_range_data(self, start_depth=None, end_depth=None):
        """
        Get data filtered by depth range
        """
        if self.depth_column is None:
            return self.data
        
        if start_depth is None:
            start_depth = self.data[self.depth_column].min()
        if end_depth is None:
            end_depth = self.data[self.depth_column].max()
        
        # Validate depth range
        start_depth = max(start_depth, self.data[self.depth_column].min())
        end_depth = min(end_depth, self.data[self.depth_column].max())
        
        mask = (self.data[self.depth_column] >= start_depth) & (self.data[self.depth_column] <= end_depth)
        return self.data[mask].copy()
    
    def calculate_pi(self, c, data=None):
        """
        Calculate Poisson Impedance for a given c value
        PI = Ip - c * Is
        """
        if data is None:
            data = self.data
        return data['Ip'] - c * data['Is']
    
    def target_correlation_analysis(self, target_var='Gr', c_range=np.arange(1.0, 3.5, 0.1), data=None):
        """
        Perform Target Correlation Coefficient Analysis (TCCA)
        """
        if data is None:
            data = self.data
            
        if target_var not in data.columns:
            return None, None, None, None
        
        correlations = []
        valid_c_values = []
        
        for c in c_range:
            try:
                pi = self.calculate_pi(c, data)
                valid_mask = ~np.isnan(pi) & ~np.isnan(data[target_var])
                if np.sum(valid_mask) > 10:
                    correlation = np.corrcoef(pi[valid_mask], data[target_var][valid_mask])[0, 1]
                    correlations.append(correlation)
                    valid_c_values.append(c)
            except:
                continue
        
        if not correlations:
            return None, None, None, None
        
        max_idx = np.argmax(np.abs(correlations))
        optimal_c = valid_c_values[max_idx]
        max_corr = correlations[max_idx]
        
        return optimal_c, max_corr, correlations, valid_c_values
    
    def find_optimal_c_values(self, start_depth=None, end_depth=None):
        """
        Find optimal c values for Lithology Impedance (LI) and Fluid Impedance (FI)
        """
        analysis_data = self.get_depth_range_data(start_depth, end_depth)
        
        # For Lithology Impedance
        lithology_indicators = ['Gr', 'Vsh', 'RT']
        li_candidates = []
        
        for indicator in lithology_indicators:
            if indicator in analysis_data.columns:
                c, corr, _, _ = self.target_correlation_analysis(indicator, data=analysis_data)
                if c is not None:
                    li_candidates.append((c, corr, indicator))
        
        if li_candidates:
            li_candidates.sort(key=lambda x: abs(x[1]), reverse=True)
            self.li_c, self.li_corr, self.li_target = li_candidates[0]
        else:
            self.li_c, self.li_corr, self.li_target = 2.0, 0.0, 'Default'
        
        # For Fluid Impedance
        fluid_indicators = ['Sw', 'RT', 'Vp']
        fi_candidates = []
        
        for indicator in fluid_indicators:
            if indicator in analysis_data.columns:
                c, corr, _, _ = self.target_correlation_analysis(indicator, data=analysis_data)
                if c is not None:
                    fi_candidates.append((c, corr, indicator))
        
        if fi_candidates:
            fi_candidates.sort(key=lambda x: abs(x[1]), reverse=True)
            self.fi_c, self.fi_corr, self.fi_target = fi_candidates[0]
        else:
            self.fi_c, self.fi_corr, self.fi_target = 1.5, 0.0, 'Default'
        
        return self.li_c, self.fi_c
    
    def calculate_impedances(self, start_depth=None, end_depth=None):
        """
        Calculate Lithology Impedance (LI) and Fluid Impedance (FI)
        """
        if not hasattr(self, 'li_c'):
            self.find_optimal_c_values(start_depth, end_depth)
        
        analysis_data = self.get_depth_range_data(start_depth, end_depth)
        
        analysis_data['LI'] = self.calculate_pi(self.li_c, analysis_data)
        analysis_data['FI'] = self.calculate_pi(self.fi_c, analysis_data)
        
        # Update main data
        if self.depth_column and (start_depth is not None or end_depth is not None):
            mask = (self.data[self.depth_column] >= analysis_data[self.depth_column].min()) & \
                   (self.data[self.depth_column] <= analysis_data[self.depth_column].max())
            self.data.loc[mask, 'LI'] = analysis_data['LI'].values
            self.data.loc[mask, 'FI'] = analysis_data['FI'].values
        else:
            self.data['LI'] = analysis_data['LI']
            self.data['FI'] = analysis_data['FI']
        
        return analysis_data[['LI', 'FI']]

# Streamlit App
def main():
    st.title("🎯 Poisson Impedance Analyzer")
    st.markdown("""
    This app performs Poisson Impedance analysis for reservoir characterization using well log data.
    Upload your CSV file with required columns (Vp, Vs, Rho) and optional columns (Gr, Sw, Vsh, RT).
    """)
    
    # Sidebar for file upload and parameters
    st.sidebar.header("📁 Data Input")
    
    uploaded_file = st.sidebar.file_uploader(
        "Upload your well log CSV file", 
        type=['csv'],
        help="Required columns: Vp, Vs, Rho. Optional: Gr, Sw, Vsh, RT, DEPTH"
    )
    
    if uploaded_file is not None:
        try:
            # Load data
            data = pd.read_csv(uploaded_file)
            
            # Initialize analyzer
            analyzer = PoissonImpedanceAnalyzer(data)
            
            # Display data info
            st.sidebar.success(f"✅ Data loaded successfully: {data.shape[0]} rows, {data.shape[1]} columns")
            
            # Depth range selection
            st.sidebar.header("🎚️ Depth Range")
            if analyzer.depth_column:
                min_depth = float(data[analyzer.depth_column].min())
                max_depth = float(data[analyzer.depth_column].max())
                
                col1, col2 = st.sidebar.columns(2)
                with col1:
                    start_depth = st.number_input(
                        "Start Depth", 
                        min_value=min_depth, 
                        max_value=max_depth, 
                        value=min_depth,
                        key="start_depth"
                    )
                with col2:
                    end_depth = st.number_input(
                        "End Depth", 
                        min_value=min_depth, 
                        max_value=max_depth, 
                        value=max_depth,
                        key="end_depth"
                    )
                
                if start_depth >= end_depth:
                    st.sidebar.error("Start depth must be less than end depth")
                    return
            else:
                st.sidebar.info("ℹ️ No depth column found. Using full dataset.")
                start_depth, end_depth = None, None
            
            # Analysis parameters
            st.sidebar.header("⚙️ Analysis Parameters")
            auto_calculate = st.sidebar.checkbox("Auto-calculate optimal c values", value=True)
            
            if not auto_calculate:
                col1, col2 = st.sidebar.columns(2)
                with col1:
                    li_c = st.number_input("LI c value", value=2.0, step=0.1)
                with col2:
                    fi_c = st.number_input("FI c value", value=1.5, step=0.1)
                analyzer.li_c, analyzer.fi_c = li_c, fi_c
            
            # Main content area
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "📊 Data Overview", 
                "🎯 TCCA Analysis", 
                "📈 Crossplots", 
                "📉 Profiles", 
                "💾 Results"
            ])
            
            with tab1:
                st.header("Data Overview")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Data Summary")
                    st.write(f"**Shape:** {data.shape}")
                    st.write(f"**Depth column:** {analyzer.depth_column if analyzer.depth_column else 'Not found'}")
                    
                    if analyzer.depth_column:
                        current_data = analyzer.get_depth_range_data(start_depth, end_depth)
                        st.write(f"**Current depth range:** {start_depth} - {end_depth}")
                        st.write(f"**Samples in range:** {len(current_data)}")
                
                with col2:
                    st.subheader("Column Statistics")
                    numeric_data = data.select_dtypes(include=[np.number])
                    st.dataframe(numeric_data.describe(), use_container_width=True)
                
                st.subheader("Data Preview")
                st.dataframe(data.head(10), use_container_width=True)
                
                # Correlation matrix
                st.subheader("Correlation Matrix")
                fig_corr = px.imshow(
                    numeric_data.corr(),
                    title="Correlation Matrix",
                    color_continuous_scale='RdBu_r',
                    aspect="auto"
                )
                st.plotly_chart(fig_corr, use_container_width=True)
            
            # Perform analysis
            if auto_calculate:
                with st.spinner("Calculating optimal c values..."):
                    li_c, fi_c = analyzer.find_optimal_c_values(start_depth, end_depth)
            else:
                li_c, fi_c = analyzer.li_c, analyzer.fi_c
            
            analyzer.calculate_impedances(start_depth, end_depth)
            current_data = analyzer.get_depth_range_data(start_depth, end_depth)
            
            with tab2:
                st.header("Target Correlation Coefficient Analysis (TCCA)")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Lithology Impedance (LI) c value", f"{li_c:.3f}")
                    st.write(f"**Target variable:** {analyzer.li_target}")
                    st.write(f"**Correlation:** {analyzer.li_corr:.3f}")
                
                with col2:
                    st.metric("Fluid Impedance (FI) c value", f"{fi_c:.3f}")
                    st.write(f"**Target variable:** {analyzer.fi_target}")
                    st.write(f"**Correlation:** {analyzer.fi_corr:.3f}")
                
                # TCCA Plots
                fig_tcca = make_subplots(
                    rows=1, cols=2,
                    subplot_titles=(
                        f'LI TCCA (vs {analyzer.li_target})',
                        f'FI TCCA (vs {analyzer.fi_target})'
                    )
                )
                
                # LI TCCA
                if hasattr(analyzer, 'li_target'):
                    _, _, li_correlations, c_range_li = analyzer.target_correlation_analysis(
                        analyzer.li_target, data=current_data)
                    if li_correlations:
                        fig_tcca.add_trace(
                            go.Scatter(x=c_range_li, y=li_correlations, 
                                      mode='lines+markers', name='LI Correlation',
                                      line=dict(color='blue', width=3),
                                      marker=dict(size=6)),
                            row=1, col=1
                        )
                        fig_tcca.add_vline(x=li_c, line_dash="dash", line_color="red", 
                                         annotation_text=f"Optimal c = {li_c:.2f}", 
                                         row=1, col=1)
                
                # FI TCCA
                if hasattr(analyzer, 'fi_target'):
                    _, _, fi_correlations, c_range_fi = analyzer.target_correlation_analysis(
                        analyzer.fi_target, data=current_data)
                    if fi_correlations:
                        fig_tcca.add_trace(
                            go.Scatter(x=c_range_fi, y=fi_correlations, 
                                      mode='lines+markers', name='FI Correlation',
                                      line=dict(color='green', width=3),
                                      marker=dict(size=6)),
                            row=1, col=2
                        )
                        fig_tcca.add_vline(x=fi_c, line_dash="dash", line_color="red", 
                                         annotation_text=f"Optimal c = {fi_c:.2f}", 
                                         row=1, col=2)
                
                fig_tcca.update_xaxes(title_text="c value", row=1, col=1)
                fig_tcca.update_xaxes(title_text="c value", row=1, col=2)
                fig_tcca.update_yaxes(title_text="Correlation Coefficient", row=1, col=1)
                fig_tcca.update_yaxes(title_text="Correlation Coefficient", row=1, col=2)
                fig_tcca.update_layout(height=500, showlegend=False)
                
                st.plotly_chart(fig_tcca, use_container_width=True)
            
            with tab3:
                st.header("Crossplots")
                
                # Determine available columns for plotting
                available_columns = current_data.columns.tolist()
                plots = []
                
                if 'Gr' in available_columns:
                    plots.append(('Gr', 'Gamma Ray', 'LI', 'Lithology Impedance (LI)', 'viridis'))
                if 'Vsh' in available_columns:
                    plots.append(('Vsh', 'Volume of Shale', 'LI', 'Lithology Impedance (LI)', 'plasma'))
                if 'Sw' in available_columns:
                    plots.append(('Sw', 'Water Saturation', 'FI', 'Fluid Impedance (FI)', 'thermal'))
                if 'RT' in available_columns:
                    plots.append(('RT', 'Resistivity', 'FI', 'Fluid Impedance (FI)', 'electric'))
                
                if plots:
                    # Create subplots
                    fig_cross = make_subplots(
                        rows=1, cols=len(plots),
                        subplot_titles=[f"{y_var} vs {x_var.split(' ')[0]}" for y_var, _, x_var, _, _ in plots]
                    )
                    
                    for idx, (col, title, imp_type, imp_name, colorscale) in enumerate(plots):
                        fig_cross.add_trace(
                            go.Scatter(
                                x=current_data[imp_type],
                                y=current_data[col],
                                mode='markers',
                                marker=dict(
                                    color=current_data[col],
                                    colorscale=colorscale,
                                    size=6,
                                    opacity=0.7,
                                    colorbar=dict(title=title)
                                ),
                                name=f'{title} vs {imp_name}',
                                hovertemplate=f'{imp_name}: %{{x:.2f}}<br>{title}: %{{y:.2f}}<extra></extra>'
                            ),
                            row=1, col=idx+1
                        )
                        
                        fig_cross.update_xaxes(title_text=imp_name, row=1, col=idx+1)
                        fig_cross.update_yaxes(title_text=title, row=1, col=idx+1)
                    
                    fig_cross.update_layout(height=500, showlegend=False)
                    st.plotly_chart(fig_cross, use_container_width=True)
                else:
                    st.info("No suitable columns found for crossplots")
                
                # 3D Crossplot
                st.subheader("3D Crossplot")
                color_var = None
                for var in ['Gr', 'Vsh', 'Sw', 'RT']:
                    if var in current_data.columns:
                        color_var = var
                        break
                
                if color_var is None:
                    color_var = 'LI'
                
                if analyzer.depth_column:
                    z_data = current_data[analyzer.depth_column]
                    z_title = analyzer.depth_column
                else:
                    z_data = current_data.index
                    z_title = 'Index'
                
                fig_3d = go.Figure(data=[
                    go.Scatter3d(
                        x=current_data['LI'],
                        y=current_data['FI'],
                        z=z_data,
                        mode='markers',
                        marker=dict(
                            size=4,
                            color=current_data[color_var],
                            colorscale='viridis',
                            opacity=0.7,
                            colorbar=dict(title=color_var)
                        ),
                        hovertemplate=(
                            f"LI: %{{x:.2f}}<br>"
                            f"FI: %{{y:.2f}}<br>"
                            f"{z_title}: %{{z:.2f}}<br>"
                            f"{color_var}: %{{marker.color:.2f}}<extra></extra>"
                        )
                    )
                ])
                
                fig_3d.update_layout(
                    scene=dict(
                        xaxis_title='Lithology Impedance (LI)',
                        yaxis_title='Fluid Impedance (FI)',
                        zaxis_title=z_title
                    ),
                    height=600
                )
                
                st.plotly_chart(fig_3d, use_container_width=True)
            
            with tab4:
                st.header("Well Log Profiles")
                
                if analyzer.depth_column:
                    depth = current_data[analyzer.depth_column]
                    depth_label = analyzer.depth_column
                else:
                    depth = current_data.index
                    depth_label = 'Sample Index'
                
                fig_profiles = make_subplots(
                    rows=1, cols=3,
                    subplot_titles=('Basic Impedances', 'Poisson Impedance Derivatives', 'Reservoir Properties'),
                    horizontal_spacing=0.08
                )
                
                # Basic impedances
                fig_profiles.add_trace(
                    go.Scatter(x=current_data['Ip'], y=depth, 
                              mode='lines', name='P-Impedance (Ip)',
                              line=dict(color='blue', width=2)),
                    row=1, col=1
                )
                fig_profiles.add_trace(
                    go.Scatter(x=current_data['Is'], y=depth, 
                              mode='lines', name='S-Impedance (Is)',
                              line=dict(color='red', width=2)),
                    row=1, col=1
                )
                
                # Derived impedances
                fig_profiles.add_trace(
                    go.Scatter(x=current_data['LI'], y=depth, 
                              mode='lines', name='Lithology Impedance (LI)',
                              line=dict(color='green', width=2)),
                    row=1, col=2
                )
                fig_profiles.add_trace(
                    go.Scatter(x=current_data['FI'], y=depth, 
                              mode='lines', name='Fluid Impedance (FI)',
                              line=dict(color='orange', width=2)),
                    row=1, col=2
                )
                
                # Reservoir properties
                if 'Gr' in current_data.columns:
                    fig_profiles.add_trace(
                        go.Scatter(x=current_data['Gr'], y=depth, 
                                  mode='lines', name='Gamma Ray',
                                  line=dict(color='black', width=2)),
                        row=1, col=3
                    )
                
                if 'Sw' in current_data.columns:
                    fig_profiles.add_trace(
                        go.Scatter(x=current_data['Sw'], y=depth, 
                                  mode='lines', name='Water Saturation',
                                  line=dict(color='purple', width=2)),
                        row=1, col=3
                    )
                
                # Update axes
                fig_profiles.update_xaxes(title_text="Impedance", row=1, col=1)
                fig_profiles.update_xaxes(title_text="Impedance", row=1, col=2)
                fig_profiles.update_xaxes(title_text="Property Value", row=1, col=3)
                
                for i in range(1, 4):
                    fig_profiles.update_yaxes(title_text=depth_label, row=1, col=i)
                    fig_profiles.update_yaxes(autorange="reversed", row=1, col=i)
                
                fig_profiles.update_layout(height=700, showlegend=True)
                st.plotly_chart(fig_profiles, use_container_width=True)
            
            with tab5:
                st.header("Results & Export")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Analysis Summary")
                    st.write(f"**Lithology Impedance (LI):** Ip - {li_c:.3f} × Is")
                    st.write(f"**Fluid Impedance (FI):** Ip - {fi_c:.3f} × Is")
                    st.write(f"**LI Correlation with {analyzer.li_target}:** {analyzer.li_corr:.3f}")
                    st.write(f"**FI Correlation with {analyzer.fi_target}:** {analyzer.fi_corr:.3f}")
                    
                    st.subheader("Statistics")
                    results_summary = current_data[['LI', 'FI']].describe()
                    st.dataframe(results_summary, use_container_width=True)
                
                with col2:
                    st.subheader("Export Results")
                    
                    # Convert results to CSV
                    csv = analyzer.data.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Full Results CSV",
                        data=csv,
                        file_name="poisson_impedance_results.csv",
                        mime="text/csv"
                    )
                    
                    # Summary statistics download
                    summary_csv = results_summary.to_csv()
                    st.download_button(
                        label="📥 Download Summary Statistics",
                        data=summary_csv,
                        file_name="poisson_impedance_summary.csv",
                        mime="text/csv"
                    )
                
                st.subheader("Processed Data Preview")
                st.dataframe(current_data.head(15), use_container_width=True)
        
        except Exception as e:
            st.error(f"Error processing data: {str(e)}")
            st.info("Please check that your CSV file contains the required columns: Vp, Vs, Rho")
    
    else:
        st.info("👆 Please upload a CSV file to get started")
        
        # Show example data structure
        st.subheader("Expected Data Format")
        example_data = pd.DataFrame({
            'DEPTH': [1000, 1001, 1002, 1003, 1004],
            'Vp': [3000, 3050, 3100, 3150, 3200],
            'Vs': [1500, 1525, 1550, 1575, 1600],
            'Rho': [2.4, 2.41, 2.42, 2.43, 2.44],
            'Gr': [45, 50, 35, 120, 130],
            'Sw': [0.2, 0.25, 0.15, 0.9, 0.95],
            'Vsh': [0.1, 0.12, 0.08, 0.8, 0.85],
            'RT': [95, 90, 100, 25, 20]
        })
        st.dataframe(example_data, use_container_width=True)
        st.caption("Note: Vp, Vs, and Rho are required. Other columns are optional but recommended for better analysis.")

if __name__ == "__main__":
    main()
