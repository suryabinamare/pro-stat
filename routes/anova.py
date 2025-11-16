from flask import Flask, request, Blueprint, render_template, current_app, jsonify
import os
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import f
from werkzeug.utils import secure_filename
import base64
import chardet
import io
import openpyxl
import xlrd
from IPython.display import display, Math



# Create a blueprint for ANOVA routes
anova_bp = Blueprint("anova", __name__, template_folder="../templates")



def allowed_files(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {"csv", "xlsx"}


@anova_bp.route("/", methods=["GET", "POST"])
def anova():
    if request.method == "GET":
        # when you open /anova in browser, show the page
        return render_template("anova.html")

    # when JavaScript sends form data (POST)
    try:
        df_num = int(request.form.get("df_num", 0))
        df_deno = int(request.form.get("df_deno", 0))
        alpha = float(request.form.get("alpha", 0.05))

        # Compute critical F value
        f_value = f.ppf(1 - alpha, df_num, df_deno)

        # Generate F-distribution curve
        mean = f.mean(df_num, df_deno)
        std = f.std(df_num, df_deno)
        x_max = mean + std * 4
        x = np.linspace(0, x_max, 500)
        y = f.pdf(x, df_num, df_deno)
        y1 = np.zeros(500)

        # Plot and save
        plt.figure(figsize=(6, 4))
        plt.plot(x, y, color="blue", label=f"F-dist (dfn={df_num}, dfd={df_deno})")
        plt.plot(x,y1)
        plt.scatter(f_value, 0, marker = 'o', color = 'red', s = 100, label=f"Critical F-value ={f_value:.3f}")
        plt.axvline(f_value, color="red", linestyle="--")
        x_fill = np.linspace(f_value, x_max, 500)
        y_fill = f.pdf(x_fill, df_num, df_deno)
        plt.fill_between(x_fill, y_fill, alpha=0.3, color="blue", label="Rejection region")
        plt.legend()
        plt.title("F-Distribution Curve")
        plt.xlabel("F-value")
        plt.ylabel("Density")
        plt.grid(True, linestyle="--", alpha=0.5)

        os.makedirs("static", exist_ok=True)
        plot_path = "static/fcurve.png"
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()

        return jsonify({
            "f_value": round(f_value, 4),
            "plot_fcurve": "/" + plot_path
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500





uploaded_df = None  # Temporary storage of DataFrame



@anova_bp.route("/upload_csv", methods=["POST"])
def upload_csv():
    global uploaded_df

    file = request.files.get("file")
    if not file:
        return jsonify({"error": "No file uploaded"}), 400

    try:
        # Optionally detect encoding
        # raw_data = file.read()
        # encoding = chardet.detect(raw_data)["encoding"]
        # file.seek(0)
        if file.filename.endswith('.xlsx') or file.filename.endswith('.xls'):
            df = pd.read_excel(file)
        else:
            df = pd.read_csv(file, encoding="utf-8")
        uploaded_df = df

        # Return HTML table of the data
        html_table = df.to_html(index=False, na_rep="", classes="fancy-table")
        return jsonify({"html": html_table}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@anova_bp.route("/get_statistics", methods=["GET"])
def get_statistics():
    global uploaded_df
    if uploaded_df is None:
        return jsonify({"error": "No file uploaded yet"}), 400

    try:
        stats_df = uploaded_df.describe().iloc[0:3]
        stats_df.iloc[0] = stats_df.iloc[0].astype(int)  # sample size as int
        stats_df.index = ['Sample size', 'Sample mean', 'Sample std.']
        stats_html = stats_df.to_html(classes="fancy-table")
        
        return jsonify({"html": stats_html}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@anova_bp.route("/get_boxplot", methods=["GET"])
def get_boxplot():
    global uploaded_df
    if uploaded_df is None:
        return jsonify({"error": "No file uploaded yet"}), 400

    try:
        x = uploaded_df.mean().values
        y = [5]*len(x)
        plt.figure(figsize=(6, 4))
        ax = plt.gca()
        grand_mean = float(np.nanmean(uploaded_df.values))
        plt.scatter(x, y, label='Sample Means')
        plt.scatter(grand_mean, [5], color='red', label='Grand Mean')
        
        
        plt.title("Sample Means and Grand Mean")
        plt.xlabel("Value")
        ax.set_yticklabels([])
        
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.5)

        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()

        return jsonify({"img_data": plot_url}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    

@anova_bp.route("/anova1", methods=["GET"])
def anova1():
    global uploaded_df
    if uploaded_df is None:
        return jsonify({"error": "No file uploaded yet"})

    try:
        numeric_cols = uploaded_df.select_dtypes(include=['number']).columns.tolist()
        if len(numeric_cols) < 2:
            return jsonify({"error": "At least two numeric columns are required for ANOVA."})

        sample_size = [int(uploaded_df[col].count()) for col in numeric_cols]
        mean_val = [float(uploaded_df[col].mean()) for col in numeric_cols]
        std_val = [float(uploaded_df[col].std(ddof=1)) for col in numeric_cols]

        flattened = uploaded_df[numeric_cols].to_numpy().flatten()
        grand_mean = float(np.nanmean(flattened))

        k = len(numeric_cols)
        N = sum(sample_size)

        SSB = sum([sample_size[i] * (mean_val[i] - grand_mean) ** 2 for i in range(k)])
        SSW = sum([(sample_size[i] - 1) * std_val[i] ** 2 for i in range(k)])

        df_between = k - 1
        df_within = N - k

        if df_within == 0:
            return jsonify({"error": "Not enough data for ANOVA."})

        MSB = SSB / df_between
        MSW = SSW / df_within

        if MSW == 0:
            return jsonify({"error": "Within-group variance is zero."})

        F = MSB / MSW
        p_value = 1 - f.cdf(F, df_between, df_within)
        anov = pd.DataFrame({
            'n':[N],
            r"$$\bar{x}$$":[grand_mean],
            'deg of numerator':[df_between],
            'deg of denominator':[df_within],
            'SSTR':[SSB],
            'SSE':[SSW],
            'MSTR':[MSB],
            'MSE':[MSW],
            'F-value':[F],
            'p-value':[p_value]

        })
        anov1 = anov.transpose().reset_index()
        anov1.columns = ['Parameter', 'Value']  

        return jsonify({
            "anova_table": anov1.to_html(index=False, float_format="%.4f", classes="fancy-table")    
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500
