function updatePlot() {
    let mean = document.getElementById("mean").value;
    let std = document.getElementById("std").value;

    document.getElementById("mean-value").innerText = mean;
    document.getElementById("std-value").innerText = std;

    fetch("/distributions/plot", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({ mean: mean, std: std })
    })
    .then(res => res.json())
    .then(data => {
        document.getElementById("plot").src =
            "data:image/png;base64," + data.image;
    });
}

// Update on slider change
document.getElementById("mean").addEventListener("input", updatePlot);
document.getElementById("std").addEventListener("input", updatePlot);

// Load initial plot
updatePlot();