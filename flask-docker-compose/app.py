from flask import Flask, jsonify, render_template
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
import json


app = Flask(__name__)


#loading main data
def load_data(file_path='data/allQuestData_all.csv'):
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        return str(e)


#loading subject metadata
def load_metadata(file_path='data/subject_data.csv'):
    try:
        metadata = pd.read_csv(file_path)
        return metadata
    except Exception as e:
        return str(e)


#section for subjects with the lowest thresholds
def compute_thresholds(df):
    exclude = ['cn', 'zh', 'jb', 'cc', 'co', 'db', 'ec', 'gc', 'jf', 'jm', 'kf']
    df_filtered = df[~df['subject'].isin(exclude)].copy()

    if "subject" not in df.columns or "threshold" not in df.columns:
        return {"error": "Missing required columns: 'subject' and 'threshold'"}
    
    #relevant subjects are grouped
    df_avg = df_filtered.groupby("subject")["threshold"].mean().reset_index()
    
    #sort average
    df_sorted = df_avg.sort_values(by="threshold", ascending=True)
    
    #pick the top 5
    lowest_thresholds = df_sorted.head(5)
    
    return lowest_thresholds


#section for average threshold of all subjects, overall and best 2 of 3
def compute_avg_thresholds(df):
    if "subject" not in df.columns or "threshold" not in df.columns or "stimulus" not in df.columns or "nStim" not in df.columns or "noise" not in df.columns:
        return {"error": "Missing required columns."}
    
    #overall average per subject, stimulus, nStim, and noise
    overall_avg = df.groupby(["subject", "stimulus", "nStim", "noise"])["threshold"].mean().reset_index()

    #best 2 out of 3 by selecting the lowest 2 values per group
    best_2_of_3 = (
        df.groupby(["subject", "stimulus", "nStim", "noise"])["threshold"]
        .apply(lambda x: x.nsmallest(2).mean())  
        .reset_index()
    )

    #final averages across subjects for each condition
    overall_avg_conditions = overall_avg.groupby(["stimulus", "nStim", "noise"])["threshold"].mean().reset_index()
    best_2_avg_conditions = best_2_of_3.groupby(["stimulus", "nStim", "noise"])["threshold"].mean().reset_index()

    return overall_avg_conditions, best_2_avg_conditions


#section for averaging thresholds per subject
def compute_subject_avg(df, subject):
    if "subject" not in df.columns or "threshold" not in df.columns or "stimulus" not in df.columns or "nStim" not in df.columns or "noise" not in df.columns:
        return {"error": "Missing required columns."}
    
    df_subject = df[df["subject"] == subject]
    
    if df_subject.empty:
        return {"error": f"No data found for subject {subject}"}

    # Compute average per condition
    avg_threshold = df_subject.groupby(["stimulus", "nStim", "noise"])["threshold"].mean().reset_index()

    # Best 2 out of 3
    best_2_of_3 = (
        df_subject.groupby(["stimulus", "nStim", "noise"])["threshold"]
        .apply(lambda x: x.nsmallest(2).mean())  
        .reset_index()
    )
    
    return avg_threshold, best_2_of_3


#handles when a subject is selected to load it, working with java in html
@app.route("/subject_plot/<subject>")
def subject_plot(subject):
    df = load_data()
    
    if isinstance(df, str):  
        return jsonify({"error": "Error loading data"}), 500
    
    subject_avg, best_2_avg = compute_subject_avg(df, subject)
    
    if isinstance(subject_avg, dict):  
        return jsonify(subject_avg), 404  # Return error if subject not found

    #generating
    plot_subject = generate_avg_plot(subject_avg, f"Thresholds for {subject}")
    plot_best_2_subject = generate_avg_plot(best_2_avg, f"Best 2 of 3 Thresholds for {subject}")
    
    return jsonify({
        "plot_subject": plot_subject,
        "plot_best_2_subject": plot_best_2_subject
    })


def categorize_visual_acuity(metadata, df):
    exclude_subjects = ['cc', 'co', 'db', 'ec', 'gc', 'jf', 'jm', 'kf']
    
    # Step 1: Filter and clean metadata
    metadata_filtered = metadata[~metadata['subject'].isin(exclude_subjects)].copy()
    metadata_filtered["visual_acuity_base"] = metadata_filtered["visual acuity"].str.extract(r"(20/\d+)").fillna("20/20")
    
    # Step 2: Categorize by acuity (numerically parsed)
    metadata_filtered['acuity_category'] = metadata_filtered['visual_acuity_base'].apply(
        lambda x: 'Lower than 20/20' if int(x.split('/')[1]) < 20 else '20/20 or higher'
    )

    # Step 3: Merge full data with subject-level acuity categories
    df_filtered = df[~df['subject'].isin(exclude_subjects)]
    merged = pd.merge(df_filtered, metadata_filtered[['subject', 'acuity_category']], on='subject', how='inner')

    # Step 4: Group by acuity + condition (stimulus, nStim, noise)
    thresholds_by_acuity = merged.groupby(['acuity_category', 'stimulus', 'nStim', 'noise'])['threshold'].mean().reset_index()

    #print(thresholds_by_acuity)

    # Step 5: Count number of subjects per category (optional table)
    acuity_counts = metadata_filtered['acuity_category'].value_counts().reset_index()
    acuity_counts.columns = ['Acuity Category', 'Subject Count']

    return acuity_counts, thresholds_by_acuity


# Section for handedness-based thresholds
def compute_thresholds_by_handedness(df, metadata):
    # Exclude the subjects
    exclude = ['cc', 'co', 'db', 'ec', 'gc', 'jf', 'jm', 'kf']
    metadata_filtered = metadata[~metadata['subject'].isin(exclude)].copy()
    df_filtered = df[~df['subject'].isin(exclude)].copy()

    # Merge metadata and data for handedness information
    merged = pd.merge(df_filtered, metadata_filtered[['subject', 'handedness']], on='subject', how='inner')
    
    # Left handed
    left_handed = merged[merged['handedness'] == 'Left']
    left_avg_threshold  = left_handed.groupby(['handedness', 'stimulus', 'nStim', 'noise'])['threshold'].mean().reset_index()

    # Right handed
    right_handed = merged[merged['handedness'] == 'Right']
    subject_averages = right_handed.groupby('subject')['threshold'].mean().reset_index()
    lowest_two_subjects = subject_averages.nsmallest(2, 'threshold')['subject']

    # Filter right-handed data for those two subjects
    right_lowest = right_handed[right_handed['subject'].isin(lowest_two_subjects)]

    # Average their thresholds per condition
    right_avg_threshold = right_lowest.groupby(['stimulus', 'nStim', 'noise'])['threshold'].mean().reset_index()

    # Handedness table count
    handedness_counts = metadata[~metadata['subject'].isin(exclude)]['handedness'].value_counts().reset_index()
    handedness_counts.columns = ['Handedness', 'Subject Count']

    return left_avg_threshold, right_avg_threshold, handedness_counts


#generating plots for everyone
def generate_avg_plot(df, title):
    plt.figure(figsize=(8, 5))

    if "stimulus" in df.columns and "nStim" in df.columns and "noise" in df.columns:
        # grouping by conditions
        # Check if we have only one noise condition for a subject
        unique_noises = df["noise"].unique()
        
        # Handle if only one noise condition is available
        if len(unique_noises) == 1:
            noise_color = "blue" if unique_noises[0] == "fixed" else "red"
            # Only one color, as the subject took part in only one noise condition
            df["condition"] = df["stimulus"] + " nStim=" + df["nStim"].astype(str)
            pivot_df = df.pivot(index="condition", columns="noise", values="threshold").fillna(0)
            bar_width = 0.4
            x = np.arange(len(pivot_df))
            plt.bar(x, pivot_df.iloc[:, 0], width=bar_width, color=noise_color, label=unique_noises[0].capitalize() + " Noise")  # Add label based on the noise condition
        else:
            # If both noise conditions are present, plot both
            df["condition"] = df["stimulus"] + " nStim=" + df["nStim"].astype(str)
            pivot_df = df.pivot(index="condition", columns="noise", values="threshold").fillna(0)
            bar_width = 0.4
            x = np.arange(len(pivot_df))
            plt.bar(x - bar_width / 2, pivot_df["fixed"], width=bar_width, color="blue", label="Fixed Noise")
            plt.bar(x + bar_width / 2, pivot_df["variable"], width=bar_width, color="red", label="Variable Noise")

        plt.xticks(ticks=x, labels=pivot_df.index, rotation=45)
        plt.legend()
    else:
        # If 'stimulus', 'nStim', 'noise' are missing, assume it's lowest_thresholds
        x_labels = df["subject"]
        plt.bar(x_labels, df["threshold"], color="black") 

    plt.title(title)
    plt.xlabel("Condition" if "condition" in df.columns else "Subject")
    plt.ylabel("Threshold")

    # Save to BytesIO
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode('utf8')


@app.route("/")
def index():
    df = load_data()
    metadata = load_metadata()
    
    #handling errors
    if isinstance(df, str) or isinstance(metadata, str):
        return jsonify({"error": "Error loading data"}), 500
    
    #getting the lowest 5 subjects
    lowest_thresholds = compute_thresholds(df)

    #getting overall and best 2 of 3
    overall_avg, best_2_avg = compute_avg_thresholds(df)

    # Generate tables for gender, handedness, age group, visual acuity
    gender_table = metadata["gender"].value_counts().reset_index()
    gender_table.columns = ["Gender", "Count"]
    
    handedness_table = metadata["handedness"].value_counts().reset_index()
    handedness_table.columns = ["Handedness", "Count"]

    left_avg_threshold, right_avg_threshold, handedness_counts = compute_thresholds_by_handedness(df, metadata)

    # Extract only the base visual acuity values (removing +/- adjustments)
    metadata["visual_acuity_base"] = metadata["visual acuity"].str.extract(r"(20/\d+)").fillna("20/20")

    # sorting visual acuity
    acuity_order = ["20/12", "20/16", "20/20", "20/25", "20/30", "20/35", "20/40"]
    metadata["visual_acuity_base"] = pd.Categorical(metadata["visual_acuity_base"], categories=acuity_order, ordered=True)

    # Count occurrences and sort
    acuity_table = metadata["visual_acuity_base"].value_counts().sort_index().reset_index()
    acuity_table.columns = ["Visual Acuity", "Count"]

    acuity_counts, thresholds_by_acuity = categorize_visual_acuity(metadata, df)

    # Categorize age into bins
    bins = [0, 20, 30, 40, 50, 60, 70, 100]
    labels = ["<20", "20-30", "30-40", "40-50", "50-60", "60-70", "70+"]
    metadata["age_group"] = pd.cut(metadata["age"], bins=bins, labels=labels, right=False)

    # Ensure correct order by making age_group categorical with an explicit order
    metadata["age_group"] = pd.Categorical(metadata["age_group"], categories=labels, ordered=True)

    # Count occurrences and sort by the categorical order
    age_table = metadata["age_group"].value_counts().sort_index().reset_index()
    age_table.columns = ["Age Group", "Count"]

    subjects = df["subject"].unique().tolist()
    
    #plotting the graphs
    plot_lowest = generate_avg_plot(lowest_thresholds, "Top 5 Subjects with Lowest Average Thresholds")
    plot_overall = generate_avg_plot(overall_avg, "Overall Average Thresholds")
    plot_best_2 = generate_avg_plot(best_2_avg, "Best 2 of 3 Average Thresholds")
    plot_low_acuity = generate_avg_plot(thresholds_by_acuity[thresholds_by_acuity['acuity_category'] == 'Lower than 20/20'], "Thresholds for Lower(Better) Acuity Subjects")
    plot_high_acuity = generate_avg_plot(thresholds_by_acuity[thresholds_by_acuity['acuity_category'] == '20/20 or higher'], "Thresholds for Higher(Worse) Acuity Subjects")
    plot_left_handed = generate_avg_plot(left_avg_threshold, "Average Thresholds for Left-Handed Subjects (2)")
    plot_right_handed = generate_avg_plot(right_avg_threshold, "Average Thresholds for Right-Handed Subjects (Best 2)")

    
    #rendering with html
    return render_template(
        "index.html",
        plot_lowest=plot_lowest,
        plot_overall=plot_overall,
        plot_best_2=plot_best_2,
        plot_low_acuity=plot_low_acuity,
        plot_high_acuity=plot_high_acuity,
        plot_left_handed=plot_left_handed,
        plot_right_handed=plot_right_handed,
        subjects=subjects,
        thresholds=lowest_thresholds.to_html(classes='table'),
        gender_table=gender_table.to_html(classes='table'),
        handedness_table=handedness_table.to_html(classes='table'),
        age_table=age_table.to_html(classes='table'),
        acuity_table=acuity_table.to_html(classes='table'),
        acuity_counts=acuity_counts.to_html(classes='table'),
        handedness_counts=handedness_counts.to_html(classes='table'),
    )




@app.route("/dashboard")
def dashboard():
    df = load_data()
    metadata = load_metadata()

    #handling errors
    if isinstance(df, str) or isinstance(metadata, str):
        return jsonify({"error": "Error loading data"}), 500
    
    #getting the lowest 5 subjects
    lowest_thresholds = compute_thresholds(df)

    #getting overall and best 2 of 3
    overall_avg, best_2_avg = compute_avg_thresholds(df)
    overall_avg_fixed = overall_avg[overall_avg['noise'] == 'fixed']
    overall_avg_variable = overall_avg[overall_avg['noise'] == 'variable']
    best_2_avg_fixed = best_2_avg[best_2_avg['noise'] == 'fixed']
    best_2_avg_variable = best_2_avg[best_2_avg['noise'] == 'variable']


    # Generate tables for gender, handedness, age group, visual acuity
    gender_table = metadata["gender"].value_counts().reset_index()
    gender_table.columns = ["Gender", "Count"]
    
    handedness_table = metadata["handedness"].value_counts().reset_index()
    handedness_table.columns = ["Handedness", "Count"]

    left_avg_threshold, right_avg_threshold, handedness_counts = compute_thresholds_by_handedness(df, metadata)

    # Extract only the base visual acuity values (removing +/- adjustments)
    metadata["visual_acuity_base"] = metadata["visual acuity"].str.extract(r"(20/\d+)").fillna("20/20")

    # sorting visual acuity
    acuity_order = ["20/12", "20/16", "20/20", "20/25", "20/30", "20/35", "20/40"]
    metadata["visual_acuity_base"] = pd.Categorical(metadata["visual_acuity_base"], categories=acuity_order, ordered=True)

    # Count occurrences and sort
    acuity_table = metadata["visual_acuity_base"].value_counts().sort_index().reset_index()
    acuity_table.columns = ["Visual Acuity", "Count"]

    acuity_counts, thresholds_by_acuity = categorize_visual_acuity(metadata, df)

    # Categorize age into bins
    bins = [0, 20, 30, 40, 50, 60, 70, 100]
    labels = ["<20", "20-30", "30-40", "40-50", "50-60", "60-70", "70+"]
    metadata["age_group"] = pd.cut(metadata["age"], bins=bins, labels=labels, right=False)

    # Ensure correct order by making age_group categorical with an explicit order
    metadata["age_group"] = pd.Categorical(metadata["age_group"], categories=labels, ordered=True)

    # Count occurrences and sort by the categorical order
    age_table = metadata["age_group"].value_counts().sort_index().reset_index()
    age_table.columns = ["Age Group", "Count"]


    subjects = df["subject"].unique().tolist()


    return render_template("dashboard.html",
                            subjects=subjects,

                            dataframes=dict(
                                # "Top 5 Subjects with Lowest Average Thresholds"
                                lowest_thresholds=lowest_thresholds.to_dict(orient="records"),
                                gender_table=gender_table.to_dict(orient="records"),
                                handedness_table=handedness_table.to_dict(orient="records"),
                                age_table=age_table.to_dict(orient="records"),
                                acuity_table=acuity_table.to_dict(orient="records"),
                                acuity_counts=acuity_counts.to_dict(orient="records"),
                                handedness_counts=handedness_counts.to_dict(orient="records"),

                                # "Overall Average Thresholds"
                                overall_avg = overall_avg.to_dict(orient="records"),
                                overall_avg_fixed = overall_avg_fixed.to_dict(orient="records"),
                                overall_avg_variable = overall_avg_variable.to_dict(orient="records"),

                                # "Best 2 of 3 Average Thresholds"
                                best_2_avg = best_2_avg.to_dict(orient="records"),
                                best_2_avg_fixed = best_2_avg_fixed.to_dict(orient="records"),
                                best_2_avg_variable = best_2_avg_variable.to_dict(orient="records"),

                                # "Thresholds for Lower(Better) Acuity Subjects"
                                low_acuity = thresholds_by_acuity[thresholds_by_acuity['acuity_category'] == 'Lower than 20/20'].to_dict(orient="records"),
                                # "Thresholds for Higher(Worse) Acuity Subjects"
                                high_acuity = thresholds_by_acuity[thresholds_by_acuity['acuity_category'] == '20/20 or higher'].to_dict(orient="records"),
                                # "Average Thresholds for Left-Handed Subjects (2)"
                                left_handed = left_avg_threshold.to_dict(orient="records"),
                                # "Average Thresholds for Right-Handed Subjects (Best 2)"
                                right_handed = right_avg_threshold.to_dict(orient="records"),

                            ),

                            tables=dict(
                                overall_avg=overall_avg.to_html(classes='table'),
                                best_2_avg=best_2_avg.to_html(classes='table'),
                                lowest_thresholds=lowest_thresholds.to_html(classes='table'),
                                gender_table=gender_table.to_html(classes='table'),
                                handedness_table=handedness_table.to_html(classes='table'),
                                age_table=age_table.to_html(classes='table'),
                                acuity_table=acuity_table.to_html(classes='table'),
                                acuity_counts=acuity_counts.to_html(classes='table'),
                                handedness_counts=handedness_counts.to_html(classes='table'),
                            )
                            )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

