document.addEventListener("DOMContentLoaded", () => {
    // ======= Element References =======
    const uploadBtn = document.getElementById("uploadBtn");
    const statsBtn = document.getElementById("statsBtn");
    const boxplotBtn = document.getElementById("boxplotBtn");
    const anovaBtn = document.getElementById("anovaBtn");
    const fileInput = document.getElementById("csvFile");
    const form = document.getElementById("anovaForm");

    let fileUploaded = false;
    // ✅ Add flags to track which outputs are already displayed
    let uploadDisplayed = false;
    let statsDisplayed = false;
    let boxplotDisplayed = false;
    let anovaDisplayed = false;
    let FValueDisplayed = false;

    // ======= Helper Functions =======

    // Remove any previously created result sections
    function clearOldResults() {
        document.querySelectorAll(".result").forEach(el => el.remove());
    }

    // Create a styled message box
    function createMessage(msg, color = "black") {
        const div = document.createElement("div");
        div.classList.add("result");
        div.innerHTML = `<p style="color:${color};">${msg}</p>`;
        return div;
    }

    // Insert a new result before a given control div
    function insertResult(controlId, contentHTML) {
        // clearOldResults();
        const controlDiv = document.getElementById(controlId);
        const newDiv = document.createElement("div");
        newDiv.classList.add("result");
        newDiv.innerHtml = "";
        newDiv.innerHTML = contentHTML;
        controlDiv.after(newDiv);
    }

    // ======= 1. Upload CSV =======
    uploadBtn.addEventListener("click", async () => {
        // clearOldResults();
        if (uploadDisplayed) return; // to check if output already displayed, in which case it outputs nothing.
        const file = fileInput.files[0];

        if (!file) {
            insertResult("controls", `<p style="color:red;">Please select a CSV file first.</p>`);
            return;
        }

        const formData = new FormData();
        formData.append("file", file);

        try {
            const response = await fetch("/anova/upload_csv", { method: "POST", body: formData });
            const result = await response.json();

            if (result.error) {
                insertResult("controls", `<p style="color:red;">Error: ${result.error}</p>`);
            } 
            
            else {
                fileUploaded = true;
                insertResult("controls", `<h2>Uploaded Data</h2>${result.html}`);
            }
        } catch (err) {
            insertResult("controls", `<p style="color:red;">Network error: ${err.message}</p>`);
        }
        uploadDisplayed = true;
    });

    // ======= 2. Get Statistics =======
    statsBtn.addEventListener("click", async () => {
        if (statsDisplayed) return;
        if (!fileUploaded) {
            insertResult("controls1", `<p style="color:red;">Please upload a CSV file first.</p>`);
            return;
        }

        try {
            const response = await fetch("/anova/get_statistics");
            const result = await response.json();

            if (result.error) {
                insertResult("controls1", `<p style="color:red;">Error: ${result.error}</p>`);
            } else {
                insertResult("controls1", `<h2>Data Statistics</h2>${result.html}`);
            }
        } catch (err) {
            insertResult("controls1", `<p style="color:red;">Network error: ${err.message}</p>`);
        }
        statsDisplayed = true;
    });

    // ======= 3. Get Boxplot =======
    boxplotBtn.addEventListener("click", async () => {
        if (boxplotDisplayed) return;
        if (!fileUploaded) {
            insertResult("controls2", `<p style="color:red;">Please upload a CSV file first.</p>`);
            return;
        
        }

        try {
            const response = await fetch("/anova/get_boxplot");
            const result = await response.json();

            if (result.error) {
                insertResult("controls2", `<p style="color:red;">Error: ${result.error}</p>`);
            } else {
                insertResult("controls2", `
                
                    <img src="data:image/png;base64,${result.img_data}" 
                         alt="Boxplot" style="max-width:100%; height:auto;">
                `);
            }
        } catch (err) {
            insertResult("controls2", `<p style="color:red;">Network error: ${err.message}</p>`);
        }
        boxplotDisplayed = true;
    });

    // ======= 4. Get ANOVA =======
    anovaBtn.addEventListener("click", async () => {
        if (anovaDisplayed) return;
        if (!fileUploaded) {
            insertResult("controls3", `<p style="color:red;">Please upload a CSV file first.</p>`);
            return;
        
        }

        try {
            const response = await fetch("/anova/anova1");
            const result = await response.json();

            if (result.error) {
                insertResult("controls3", `<p style="color:red;">Error: ${result.error}</p>`);
            } else {
                insertResult("controls3", `
                    <h2>ANOVA Results</h2>
                    <p>${result["anova_table"]}</p>`);
            }
        } catch (err) {
            insertResult("controls3", `<p style="color:red;">Network error: ${err.message}</p>`);
        }
        anovaDisplayed = true;
    });

   // .............Get F-value............ 
form.addEventListener("submit", async (event) => {
     event.preventDefault(); // ✅ prevents the form from doing a page reload

    // ✅ Collect all input values (df_num, df_deno, alpha)
    const formData = new FormData(form);

    try {
        // ✅ Send to Flask route ("/anova") using POST
        const response = await fetch(form.action, {
            method: "POST",
            body: formData,
        });

        // ✅ Expect a JSON response from Flask
        const data = await response.json();

        // ✅ Display the returned F-value
        const resultDiv = document.getElementById("result");
        resultDiv.innerHTML = `
            <p> <b>Critical F-Value: ${data.f_value}</b></p>
            <img src="${data.plot_fcurve}" alt="F-distribution Plot" width="500"
            style="max-width:100%; height:auto;">
        `;

    } catch (error) {
        resultDiv.innerHTML = ("Error sending the data:", result.error);
    }
    });

});
