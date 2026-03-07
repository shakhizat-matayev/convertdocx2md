import pandas as pd
import os

# Define paths
input_dir = 'data_csv'
output_dir = 'data_csv/md_profiles'

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Load the dataframes
master_df = pd.read_csv(os.path.join(input_dir, 'Data_1_Master_Table.csv'))
seismic_df = pd.read_csv(os.path.join(input_dir, 'Data_6_4D_Seismic.csv'))

# Clean Well IDs (Force Uppercase and strip whitespace for perfect joining)
master_df['Well_ID'] = master_df['Well_ID'].astype(str).str.upper().str.strip()
seismic_df['Well'] = seismic_df['Well'].astype(str).str.upper().str.strip()

# Get unique list of wells from the Master Table
all_wells = master_df['Well_ID'].unique()

for well in all_wells:
    if well == 'NAN': continue  # Skip empty rows if any
    
    # Filter data for this specific well
    m_row = master_df[master_df['Well_ID'] == well].iloc[0]
    s_rows = seismic_df[seismic_df['Well'] == well]

    # Construct the Rich Markdown Content
    content = f"# Well Health Profile: {well}\n\n"
    
    content += "## 1. General Well & Reservoir Data\n"
    content += f"- **Platform Location**: {m_row['Location']}\n"
    content += f"- **Development Polygon**: {m_row['Polygon_Name']}\n"
    content += f"- **Reservoir Name**: {m_row['Reservoir']}\n"
    content += f"- **Well Classification**: {'Oil Producer (OP)' if m_row['Well_Type'] == 'OP' else 'Water Injector (WI)'}\n\n"

    content += "## 2. Production & Injection Metrics (TMORE)\n"
    content += f"- **Cumulative Oil Production**: {m_row['TMORE_Allocation_Oil Cummulative']} barrels\n"
    content += f"- **Latest Daily Production Rate**: {m_row['TMORE_Allocation_OP Daily']} barrels/day\n"
    content += f"- **Latest Water Injection Rate**: {m_row['TMORE_Allocation_Winj']}\n"
    content += f"- **Latest Watercut (WCT)**: {m_row['TMORE_WCT']}\n\n"

    content += "## 3. Petrophysical Log Data (Petrel)\n"
    content += f"- **Average Porosity (Phie)**: {m_row['Log_Avg_Phie']}\n"
    content += f"- **Average Water Saturation (Sw)**: {m_row['Log_Avg_Sw']}\n"
    content += f"- **Frac Density**: {m_row['Log_Avg_Frac']}\n"
    content += f"- **Average Permeability**: {m_row['Log_Avg_Perm']} mD\n\n"

    content += "## 4. Manual Expert Assessments\n"
    content += f"- **Performance Analysis**: {m_row['Manual Performance Analysis']}\n"
    content += f"- **Seismic 4D Response**: {m_row['Manual_Seismic 4D Response']}\n\n"

    content += "## 5. AI/ML Analytics & Guard Outputs\n"
    content += f"- **FlowGuard (Performance Class)**: {m_row['FlowGuard_Perf Class']}\n"
    content += f"- **GeaGuard (Seismic Class)**: {m_row['GeaGuard_4D Reponse_Class']}\n"
    content += f"- **StimAtlas (Stimulation Recommendation)**: {m_row['StimAtlas_Recommendation']}\n"
    content += f"- **FracGuard (Fracture Event Risk)**: {m_row['FracGuard_Frac Risk']}\n"
    content += f"- **TerraGuard (Losses/Fault Risk)**: {m_row['TerraGuard_Frac & Losses Risk']}%\n"
    content += f"- **WellGuard (Non-Conformance Risk)**: {m_row['WellGuard_Non Conf Risk']}\n"
    content += f"- **SweepGuard (VRR Class)**: {m_row['SweepGuard_VRR']}\n"
    content += f"- **TestGuard (Test Priority)**: {m_row['TestGuard_Priority']}\n\n"

    # Add Detailed 4D Seismic Signals if they exist in Data_6
    if not s_rows.empty:
        content += "## 6. Detailed 4D Seismic Interpretation\n"
        for _, s_row in s_rows.iterrows():
            content += f"### Vintage: {s_row['Vintage']}\n"
            content += f"- **Signal**: {s_row['Signal']}\n"
            content += f"- **Interpretation**: {s_row['Signal interpretation / dynamic explanation']}\n"
            content += f"- **Remarks**: {s_row['Remarks']}\n\n"

    # Write to file (Overwrites automatically)
    file_path = os.path.join(output_dir, f"Well_Profile_{well}.md")
    with open(file_path, "w") as f:
        f.write(content)

print(f"Success: Processed {len(all_wells)} well profiles into {output_dir}")