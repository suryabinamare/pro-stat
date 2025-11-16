from flask import Flask, request, Blueprint, render_template, current_app
import os
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import f
from scipy.stats import norm
from scipy.stats import t
from scipy.stats import chi2
from werkzeug.utils import secure_filename


chisquare_bp = Blueprint("chisquare", __name__)

@chisquare_bp.route("/goodnesstest", methods=["GET", "POST"])
def goodnesstest():
    values = None
    if request.method == "POST":
        file = request.files["goodness_test"]
        if not file:
            return "No file uploaded", 400
        #read csv or excel
        if file.filename.endswith(".csv"):
            df = pd.read_csv(file)
        elif file.filename.endswith((".xls", ".xlsx")):
            df = pd.read_excel(file)
        else:
            return "Invalid file format. Please upload CSV or XLSX.", 400
        #check for required columns
        df.columns = df.columns.map(str.lower)

        if 'observed' not in df.columns or 'expected' not in df.columns:
            return "CSV must contain 'observed' and 'expected' columns.", 400
        observed = df['observed'].values
        expected = df['expected'].values
        stats = (observed - expected) ** 2 / expected
        df['(expected-observed)^2/expected'] = stats
        chi2_stat = np.sum(stats)
        p_value = chi2.sf(chi2_stat, df=len(observed)-1)
        alpha = request.form.get("alpha", type=float)
        chi2_crit = chi2.isf(alpha, df=len(observed)-1)
        values = [chi2_stat, p_value, chi2_crit, alpha, len(observed)-1]
        table_html = df.to_html()

    return render_template("goodnesstest.html", title="Goodness of Fit Test", values=values, table_html=table_html if values else None)




@chisquare_bp.route("/independencetest", methods=["GET", "POST"])
def independencetest():
    values = None
    df_html = None
    df1_html = None
    df_diff_html = None
    if request.method == "POST":
        file = request.files["independence_test"]
        if not file:
            return "No file uploaded", 400
        #read csv or excel
        if file.filename.endswith(".csv"):
            df = pd.read_csv(file)
        elif file.filename.endswith((".xls", ".xlsx")):
            df = pd.read_excel(file)
        else:
            return "Invalid file format. Please upload CSV or XLSX.", 400
        df = df.set_index(df.columns[0])
        df_normalized = df.div(df.sum(axis=1), axis=0) * 100
      


      # Calculate expected frequencies
        sum = df.values.sum()
        row_sums = df.sum(axis = 1).to_list()
        col_sums = df.sum(axis = 0).to_list()
        df_expected = np.outer(row_sums, col_sums) / sum
        df1 = pd.DataFrame(df_expected, columns = df.columns, index = df.index)
        df1_normalized = df1.div(df1.sum(axis=1), axis=0)* 100

        df_diff = (df1 - df) ** 2 / df1
        chi2_stat = df_diff.values.sum()
        dof = (len(row_sums)-1) * (len(col_sums)-1)

        p_value = chi2.sf(chi2_stat, df=dof)
        alpha = request.form.get("alpha", type=float)
        if alpha>=1 or alpha<0:
            return "Invalid alpha value. Please enter a value between 0 and 1.", 400
        else:
            chi2_crit = chi2.isf(alpha, df=dof)
        values = [chi2_stat, p_value, chi2_crit, alpha, dof]
        DF = df.reset_index()
        DF.rename(columns={DF.columns[0]:" "}, inplace=True)
        DF1 = df1.reset_index()
        DF1.rename(columns={DF1.columns[0]:" "}, inplace=True)
        df_html = DF.to_html(classes="table table-bordered table-striped table-hover", index=False)
        df1_html = DF1.to_html(classes="table table-bordered table-striped table-hover", index=False)
        df_diff_html = df_diff.to_html(classes="table table-bordered table-striped table-hover", index=False)



        #side by side bar graph of observed vs expected frequencies:
        fig1, ax3 = plt.subplots(figsize=(8, 6))
        width1 = 0.35 
        x1 = range(len(df.index))
        for i, col in enumerate(df.columns):
            ax3.bar(
                [pos + i * width1 for pos in x1], #shift bars for side-by-side
                df[col],
                width = width1,
                label = f"{col} (Table 1)" if i ==0 else "",
                alpha = 0.7)
            ax3.bar(
                [pos + i * width1 for pos in x1],
                df1[col],
                width = width1,
                label = f"{col} (Table 2)" if i ==0 else "",
                alpha = 0.7)
        ax3.set_xticks([pos + width1/2 for pos in x1])
        ax3.set_xticklabels(df.index, rotation=45)
        ax3.set_title("Observed vs Expected Frequencies")
        ax3.set_ylabel("Frequency")
        ax3.set_xlabel("Categories")
        ax3.legend()
        plt.tight_layout()
        plt.savefig("static/observed_vs_expected.png")
        plt.close()




        #plots: observed vs expected (relative)
        # plots: observed vs expected
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        x = np.arange(len(df.index))
        width = 0.5

        # ------------------
        # Observed plot
        bottoms = np.zeros(len(df_normalized.index))
        for col in df_normalized.columns:
            col_values = df_normalized[col].values
            bars = ax1.bar(x, col_values, width, bottom=bottoms, label=col)
            for bar, val, bottom in zip(bars, col_values, bottoms):
                if val > 0:
                    ax1.text(bar.get_x() + bar.get_width()/2,
                            bottom + val/2,
                            f"{val:.2f}%",
                            ha='center', va='center', color='white', fontsize=10)
            bottoms += col_values

        ax1.set_xticks(x)
        ax1.set_xticklabels(df.index, rotation=45)
        ax1.set_title("Observed Relative Frequencies")
        ax1.set_ylabel("Percentage")
        ax1.set_xlabel("Categories")

        # ------------------
        # Expected plot
        bottoms = np.zeros(len(df1_normalized.index))  # reset for second plot
        for col in df1_normalized.columns:
            col_values = df1_normalized[col].values
            bars = ax2.bar(x, col_values, width, bottom=bottoms, label=col)
            for bar, val, bottom in zip(bars, col_values, bottoms):
                if val > 0:
                    ax2.text(bar.get_x() + bar.get_width()/2,
                            bottom + val/2,
                            f"{val:.2f}%",
                            ha='center', va='center', color='white', fontsize=10)
            bottoms += col_values

        ax2.set_xticks(x)
        ax2.set_xticklabels(df1.index, rotation=45)
        ax2.set_title("Expected Relative Frequencies")
        ax2.set_ylabel("Percentage")
        ax2.set_xlabel("Categories")

        # ------------------
        # Single legend for the figure
        handles, labels = ax1.get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=len(labels))

        # Adjust layout so legend doesn’t overlap
        plt.tight_layout(rect=[0, 0, 1, 0.9])

        # save the figure
        plt.savefig("static/expected_values.png")
        plt.close()









    return render_template("independencetest.html", title="Independence Test", 
                           values=values, df_html=df_html, 
                           df1_html=df1_html, 
                           df_diff_html=df_diff_html)

