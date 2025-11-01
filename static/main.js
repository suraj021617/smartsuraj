document.addEventListener("DOMContentLoaded", function () {
    const initData = JSON.parse(document.getElementById('init-data').textContent);
    const { dates, default_date } = initData;
    const datePicker = document.getElementById("datePicker");
    const searchBtn = document.getElementById("searchBtn");

    let selectedDate = "";
    flatpickr(datePicker, {
        dateFormat: "Y/m/d",
        defaultDate: default_date ? default_date.replace(/-/g, "/") : null,
        allowInput: false,
        enable: dates.map(dt => dt.replace(/-/g, "/")),
        onChange: function (selectedDates, dateStr) {
            selectedDate = dateStr;
        }
    });
    if (default_date) {
        datePicker.value = default_date.replace(/-/g, "/");
        selectedDate = datePicker.value;
    }

    function yyyymmdd_with_slash_to_yyyymmdd_dash(val) {
        return val ? val.replace(/\//g, "-") : "";
    }

    function fetchResults() {
        let date = selectedDate || datePicker.value;
        date = yyyymmdd_with_slash_to_yyyymmdd_dash(date);
        if (!date) return;
        fetch(`/get_results?date=${date}`)
            .then(res => res.json())
            .then(res => {
                const container = document.getElementById("resultsCard");
                const patternArea = document.getElementById("patternArea");
                patternArea.innerHTML = ""; // clear
                if (!res.success || !res.data.length) {
                    container.innerHTML = "<div class='text-center text-gray-500'>No result for this date.</div>";
                    return;
                }
                container.innerHTML = res.data.map((d, i) => `
                    <div class="border border-gray-300 rounded-md shadow-sm w-[320px] p-4 mb-4 bg-white">
                        <div class="font-semibold text-lg mb-2">${d.provider || ""}</div>
                        <div class="text-sm text-gray-600 mb-1">${d.draw_info || ""}</div>
                        <div><b>1st:</b> ${d.first || "-"} <b>2nd:</b> ${d.second || "-"} <b>3rd:</b> ${d.third || "-"}</div>
                        <div class="mt-2">
                            <b>Special:</b>
                            ${(d.special && d.special.length) ? d.special.map(n => `<span class="inline-block bg-gray-100 px-2 mx-1">${n}</span>`).join('') : "-"}
                        </div>
                        <div class="mt-2">
                            <b>Consolation:</b>
                            ${(d.consolation && d.consolation.length) ? d.consolation.map(n => `<span class="inline-block bg-gray-100 px-2 mx-1">${n}</span>`).join('') : "-"}
                        </div>
                        <div class="mt-4 text-right">
                            <button class="pattern-btn bg-green-600 text-white px-3 py-1 rounded" data-index="${i}">Check Patterns</button>
                        </div>
                    </div>
                `).join('');

                // Attach event for pattern check
                Array.from(document.querySelectorAll('.pattern-btn')).forEach(btn => {
                    btn.onclick = function () {
                        const d = res.data[parseInt(btn.dataset.index)];
                        showPatternFinder(d);
                    };
                });
            });
    }

    function showPatternFinder(d) {
        // Sample: combine all numbers into a single string, show as 4x4 grid
        let nums = [d.first, d.second, d.third, ...d.special, ...d.consolation].filter(Boolean);
        let flat = nums.join("").replace(/[^0-9]/g, "").slice(0, 16);
        // Build 4x4 grid
        let grid = [];
        for (let i = 0; i < 16; i += 4) {
            grid.push(flat.slice(i, i + 4).split(""));
        }
        let gridHtml = `<div class="mb-2 text-center font-bold">4x4 Pattern Grid:</div>
        <div class="flex flex-col items-center gap-1 mb-4">
            ${grid.map(row =>
                `<div class="flex gap-2">${row.map(cell => `<span class="w-8 h-8 flex items-center justify-center bg-gray-200 rounded text-lg">${cell || ""}</span>`).join('')}</div>`
            ).join('')}
        </div>
        <div class="mb-2 text-center"><button class="analyze-btn bg-blue-700 text-white px-4 py-1 rounded">Analyze Pattern</button></div>
        <div id="patternResult" class="text-center text-green-700 font-semibold mt-3"></div>
        `;
        const patternArea = document.getElementById("patternArea");
        patternArea.innerHTML = gridHtml;

        // Pattern finding (dummy for now)
        document.querySelector('.analyze-btn').onclick = function () {
            // TODO: Real pattern check
            document.getElementById("patternResult").textContent = "Pattern detected: (demo only — add real logic)";
        };
    }

    fetchResults();
    searchBtn.onclick = fetchResults;
});
