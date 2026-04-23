import json
import numpy as np

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;600;800&family=Inter:wght@400;500&display=swap');
:root {
    --bg-color: #0b0f19; --card-bg: rgba(20, 27, 45, 0.6);
    --accent: #6d28d9; --accent-hover: #7c3aed;
    --success: #2dd4bf; --text-main: #f8fafc; --text-muted: #94a3b8;
}
body {
    font-family: 'Inter', sans-serif; background-color: var(--bg-color); color: var(--text-main); margin: 0; min-height: 100vh; overflow-x: hidden;
}
h1, h2, h3, h4 { font-family: 'Outfit', sans-serif; }
.container { padding: 40px; max-width: 1400px; margin: 0 auto; }
.hero { text-align: center; margin-bottom: 50px; }
.title { font-size: 2.5rem; font-weight: 800; background: linear-gradient(to right, #a78bfa, #2dd4bf); background-clip: text; -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.tagline { color: var(--text-muted); font-size: 1.2rem; }
.badges { display: flex; justify-content: center; gap: 20px; margin-top: 20px; }
.badge { background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1); padding: 15px 25px; border-radius: 12px; }
.badge-val { font-size: 1.8rem; font-weight: bold; color: var(--success); }
.badge-lbl { font-size: 0.9rem; color: var(--text-muted); text-transform: uppercase; }

.section-title { font-size: 1.8rem; border-bottom: 2px solid rgba(255,255,255,0.1); padding-bottom: 10px; margin-top: 50px; }
.card { background: var(--card-bg); backdrop-filter: blur(12px); border: 1px solid rgba(255,255,255,0.05); border-radius: 16px; padding: 30px; margin-bottom: 40px; }
.grid-3 { display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; }
.grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
.grid-4 { display: grid; grid-template-columns: 1fr 50px 1fr 50px 1fr 50px 1fr; gap: 10px; align-items: center; }

.flow-step { text-align: center; padding: 20px; background: rgba(0,0,0,0.3); border-radius: 8px; border: 1px solid var(--accent); }
.flow-arrow { display: flex; align-items: center; justify-content: center; font-size: 2rem; color: var(--accent); }

table { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
th, td { padding: 8px 10px; text-align: left; border-bottom: 1px solid rgba(255,255,255,0.05); white-space: nowrap; }
th { color: #cbd5e1; background: rgba(255,255,255,0.03); }
tr:hover { background: rgba(255,255,255,0.03); }
.table-responsive { overflow-x: auto; background: rgba(0,0,0,0.2); border-radius: 8px; margin-bottom: 20px; }
.fplpa-col { color: var(--success) !important; font-weight: bold; background: rgba(45, 212, 191, 0.05); }

.calculator { background: rgba(109,40,217,0.1); border: 1px solid var(--accent); padding: 20px; border-radius: 12px; margin-top: 15px; display: flex; align-items: center; gap: 15px; }
select, button { padding: 8px 15px; border-radius: 6px; background: rgba(255,255,255,0.1); color: white; border: 1px solid rgba(255,255,255,0.2); }
option { background: #0b0f19; color: white; }
button { cursor: pointer; background: var(--accent); font-weight: bold; }
button:hover { background: var(--accent-hover); }

.chart-container { position: relative; height: 350px; width: 100%; background: #1a202c; border-radius: 8px; padding: 15px; box-sizing: border-box; }
.footer { text-align: left; padding: 40px; color: var(--text-muted); border-top: 1px solid rgba(255,255,255,0.05); margin-top: 50px; line-height: 1.6; font-size: 0.9rem; background: rgba(0,0,0,0.2); border-radius: 12px;}

.tab-container { margin-bottom: 20px; display: flex; gap: 10px; border-bottom: 2px solid rgba(255,255,255,0.1); padding-bottom: 10px; }
.tab-btn { background: transparent; border: none; color: var(--text-muted); font-size: 1.1rem; padding: 10px 20px; cursor: pointer; border-radius: 8px 8px 0 0; transition: 0.3s; }
.tab-btn:hover { color: white; background: rgba(255,255,255,0.05); }
.tab-btn.active { color: var(--accent); border-bottom: 3px solid var(--accent); background: rgba(109,40,217,0.1); }
.tab-content { display: none; animation: fadeIn 0.5s; }
.tab-content.active { display: block; }
@keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }

.peak-badge { background: rgba(245, 158, 11, 0.2); color: #fcd34d; font-size: 0.75rem; padding: 2px 6px; border-radius: 4px; border: 1px solid rgba(245, 158, 11, 0.4); margin-left: 8px; font-style: normal; white-space: nowrap; }
.gain-pos { color: #10b981; font-weight: bold; }
.gain-neg { color: #ef4444; font-weight: bold; }

.slider-container { margin-top: 20px; background: rgba(0,0,0,0.2); padding: 15px; border-radius: 8px; display: flex; align-items: center; gap: 15px;}
.slider-container label { font-weight: bold; color: #a78bfa; width: 150px;}
.slider { flex: 1; cursor: pointer; accent-color: var(--accent); }

#toast { visibility: hidden; min-width: 250px; background-color: rgba(16, 185, 129, 0.9); color: #fff; text-align: center; border-radius: 8px; padding: 16px; position: fixed; z-index: 1000; left: 50%; bottom: 30px; transform: translateX(-50%); opacity: 0; transition: opacity 0.5s, bottom 0.5s; backdrop-filter: blur(10px); }
#toast.show { visibility: visible; opacity: 1; bottom: 50px; }

.share-btn { float: right; background: rgba(255,255,255,0.1); color: white; border: 1px solid rgba(255,255,255,0.2); padding: 8px 15px; border-radius: 6px; cursor: pointer; margin-top: -55px; }
.share-btn:hover { background: rgba(255,255,255,0.2); }

</style>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
"""

def build_chartjs_script(sim, chart_index):
    clients = list(sim["history_auc"].keys())
    rounds = sim["rounds"]
    labels = json.dumps([f"Round {r}" for r in range(1, rounds + 1)])
    
    auc_datasets = []
    gmean_datasets = []
    colors = ['#8b5cf6', '#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#ec4899', '#14b8a6', '#f97316']
    
def build_chartjs_script(history_dict, title, chart_id, metric_name, color_map):
    labels = [f"Round {i+1}" for i in range(len(list(history_dict.values())[0]))]
    datasets = []
    for cid, vals in history_dict.items():
        datasets.append({"label": cid, "data": vals, "borderColor": color_map[cid], "fill": False, "tension": 0.4})
    
    html = f"""
    <script>
    window['og_data_{chart_id}'] = {json.dumps(datasets)};
    window['globalLabels'] = {json.dumps(labels)};
    var chart_{chart_id.replace('-','_')} = new Chart(document.getElementById('{chart_id}'), {{
        type: 'line', data: {{ labels: {json.dumps(labels)}, datasets: JSON.parse(JSON.stringify(window['og_data_{chart_id}'])) }},
        options: {{ responsive: true, maintainAspectRatio: false, plugins: {{ title: {{ display: true, text: '{title}', color: '#fff' }}, legend: {{ labels: {{ color: '#fff' }} }} }}, scales: {{ y: {{ ticks: {{ color: '#fff' }} }}, x: {{ ticks: {{ color: '#fff' }} }} }} }}
    }});
    </script>
    """
    return html

def build_bar_chart(sim, chart_id):
    baselines_list = sim["baselines"] + ["FPLPA"]
    clients = list(sim["history_auc"].keys())
    
    auc_avgs = []
    gm_avgs = []
    
    for b in baselines_list:
        if b == "FPLPA":
            auc_avgs.append(float(np.mean([sim["history_auc"][c][-1] for c in clients])))
            gm_avgs.append(float(np.mean([sim["history_gmean"][c][-1] for c in clients])))
        else:
            auc_avgs.append(float(np.mean([sim["collected_b_aucs"][idx][b] for idx, c in enumerate(clients)])))
            gm_avgs.append(float(np.mean([sim["collected_b_gmeans"][idx][b] for idx, c in enumerate(clients)])))
            
    datasets = [
        {"label": "Avg AUC", "data": auc_avgs, "backgroundColor": "#38bdf8"},
        {"label": "Avg G-Mean", "data": gm_avgs, "backgroundColor": "#f43f5e"}
    ]
    
    script = f"""
    <script>
    new Chart(document.getElementById('{chart_id}'), {{
        type: 'bar',
        data: {{ labels: {json.dumps(baselines_list)}, datasets: {json.dumps(datasets)} }},
        options: {{ responsive: true, maintainAspectRatio: false, plugins: {{ title: {{ display: true, text: 'Average Score Comparison', color: '#fff' }}, legend: {{ labels: {{ color: '#fff' }} }} }}, scales: {{ y: {{ ticks: {{ color: '#fff' }} }}, x: {{ ticks: {{ color: '#fff' }} }} }} }}
    }});
    </script>
    """
    return script

def build_hero(results, config):
    all_aucs = []
    all_gms = []
    total_clients = 0
    for r in results:
        total_clients += len(r["history_auc"].keys())
        for cid in r["history_auc"]:
            all_aucs.append(r["history_auc"][cid][-1])
            all_gms.append(r["history_gmean"][cid][-1])
            
    best_auc = max(all_aucs) if all_aucs else 0
    best_gm = max(all_gms) if all_gms else 0
    
    return f"""
    <div class='hero'>
        <h1 class='title'>FPLPA Dashboard</h1>
        <p class='tagline'>Federated Prototype Learning for Heterogeneous Software Defect Prediction</p>
        <div class='badges'>
            <div class='badge'><div class='badge-val'>{best_auc:.4f}</div><div class='badge-lbl'>Peak AUC</div></div>
            <div class='badge'><div class='badge-val'>{best_gm:.4f}</div><div class='badge-lbl'>Peak G-Mean</div></div>
            <div class='badge'><div class='badge-val'>{total_clients}</div><div class='badge-lbl'>Clients Evaluated</div></div>
        </div>
    </div>
    """

def build_process_flow():
    return """
    <h2 class='section-title'>How It Works</h2>
    <div class='grid-4'>
        <div class='flow-step'><strong>1. Data Preprocessing</strong><br><small>OSS + Chi-Square</small></div>
        <div class='flow-arrow'>→</div>
        <div class='flow-step'><strong>2. Local CPN Training</strong><br><small>Extracting Features</small></div>
        <div class='flow-arrow'>→</div>
        <div class='flow-step'><strong>3. Prototype Aggregation</strong><br><small>Privacy Configured Server</small></div>
        <div class='flow-arrow'>→</div>
        <div class='flow-step'><strong>4. Personalization</strong><br><small>Local Execution</small></div>
    </div>
    <div class='card' style='margin-top: 20px; background: rgba(16, 185, 129, 0.1); border-color: #10b981;'>
        <h3 style='color: #10b981; margin-top: 0;'>Secure Privacy Protection</h3>
        <p>This architecture inherently protects sensitive corporate intellectual property. By aggregating strictly <strong>Prototypes</strong> instead of raw code parameters, external hosts cannot reverse-engineer your project source code.</p>
    </div>
    """

def build_config_cards(config):
    return f"""
    <div class='grid-3'>
        <div class='card' style='margin-bottom: 0;'>
            <h3 style='margin-top: 0; color: #a78bfa;'>Datasets Included</h3>
            <ul style='color: var(--text-muted); line-height: 1.8;'>
                <li><strong>NASA MDP:</strong> 12 C++ Projects</li>
                <li><strong>AEEEM:</strong> 5 Java Projects</li>
                <li><strong>Relink:</strong> 3 Java Projects</li>
            </ul>
        </div>
        <div class='card' style='margin-bottom: 0;'>
            <h3 style='margin-top: 0; color: #a78bfa;'>Local Hyperparameters</h3>
            <ul style='color: var(--text-muted); line-height: 1.8;'>
                <li><strong>Learning Rate (LR):</strong> {config.get('lr', 0.01)}</li>
                <li><strong>Client Epochs:</strong> {config.get('epochs', 5)}</li>
                <li><strong>Regularization (λ):</strong> {config.get('lam', 0.1)}</li>
            </ul>
        </div>
        <div class='card' style='margin-bottom: 0;'>
            <h3 style='margin-top: 0; color: #a78bfa;'>Server Hyperparameters</h3>
            <ul style='color: var(--text-muted); line-height: 1.8;'>
                <li><strong>Comm. Rounds:</strong> {config.get('rounds', 20)}</li>
                <li><strong>Feature Space:</strong> 25 Chi2 Dimensions</li>
                <li><strong>Target Clients:</strong> {config.get('subset', 3)} sampled per round</li>
            </ul>
        </div>
    </div>
    """

def generate_dashboard(results, config, output_file="dashboard.html"):
    html = f"<!DOCTYPE html><html><head><title>FPLPA Secure Dashboard</title>{CSS}</head><body>"
    html += "<div class='container'>"
    html += build_hero(results, config)
    html += build_process_flow()
    
    html += "<h2 class='section-title'>System Configuration</h2>"
    html += build_config_cards(config)
    
    globals = [r for r in results if r["type"] == "global"]
    baselines = [r for r in results if r["type"] == "baseline"]
    
    html += "<h2 class='section-title'>Part 1: Global Cross-Project Progress</h2>"
    html += "<div class='tab-container'>"
    for i, sim in enumerate(globals):
        title = sim['pair_name'].replace('_', ' ')
        active = "active" if i == 0 else ""
        html += f"<button class='tab-btn tab-g-btn {active}' id='tab-g-btn-{i}' onclick='switchTab(\"tab-g\", {i})'>{title}</button>"
    html += "</div>"
    
    for i, sim in enumerate(globals):
        title = sim['pair_name'].replace('_', ' ')
        html += build_eval_matrix_table(sim, f"table-g-{i}")
        html += "<div class='grid-2'>"
        html += f"<div class='chart-container'><canvas id='auc-chart-g-{i}'></canvas></div>"
        html += f"<div class='chart-container'><canvas id='gm-chart-g-{i}'></canvas></div>"
        html += "</div>"
        
        clients = list(sim["history_auc"].keys())
        color_map = {}
        colors = ["#8b5cf6", "#3b82f6", "#10b981", "#f59e0b", "#ef4444", "#ec4899", "#14b8a6", "#f97316"]
        for j, cid in enumerate(clients): color_map[cid] = colors[j % len(colors)]
            
        html += build_chartjs_script(sim["history_auc"], "AUC Convergence", f"auc-chart-g-{i}", "AUC", color_map)
        html += build_chartjs_script(sim["history_gmean"], "G-Mean Convergence", f"gm-chart-g-{i}", "G-mean", color_map)
        html += "</div>" # close card
        
    html += "<h2 class='section-title'>Part 2: Baseline Algorithm Comparison</h2>"
    base_options = "<select id='baseline-select' onchange='calcImprovement()'>"
    if len(baselines) > 0 and len(baselines[0]["baselines"]) > 0:
        for b in baselines[0]["baselines"]: base_options += f"<option value='{b}'>{b}</option>"
    base_options += "</select>"
    
    html += f"<div class='calculator'><strong>Improvement Calculator: </strong> Compare FPLPA against: {base_options} <button onclick='calcImprovement()'>Compute</button><span id='calc-result'></span></div><br>"
    
    html += "<div class='tab-container'>"
    for i, sim in enumerate(baselines):
        title = sim['pair_name'].replace('_', ' ')
        active = "active" if i == 0 else ""
        html += f"<button class='tab-btn tab-b-btn {active}' id='tab-b-btn-{i}' onclick='switchTab(\"tab-b\", {i})'>{title}</button>"
    html += "</div>"
    
    for i, sim in enumerate(baselines):
        html += build_baseline_table(sim, f"table-b-{i}")
        html += f"<div class='chart-container' style='height: 400px;'><canvas id='bar-chart-b-{i}'></canvas></div>"
        
        # Build bar chart payload
        html += build_bar_chart(sim, f"bar-chart-b-{i}")
        
        html += "</div>" # close card
        
    html += f"""
    <div class='footer'>
        <button class='share-btn' onclick='copyShareLink()'>🔗 Share Results</button>
        <h3 style="margin-top:0;">References & Research Footnotes</h3>
        <p><strong>Datasets:</strong> Software defect prediction datasets obtained from NASA MDP, AEEEM, and Relink repositories. Features are aligned using Chi-Square secure reduction.</p>
        <p><strong>FPLPA Target Citation:</strong> Proposed Federated Prototype Learning paradigm corresponding to framework methodologies outlined in current AI / Software Engineering research.</p>
        <p><strong>Baseline Methods:</strong> Evaluated against standard privacy-preserving and transfer learning paradigms including <em>FedAvg</em> (McMahan et al.), <em>FRLGC</em>, and <em>FTLKD</em>.</p>
    </div>
    """
    html += f"</div>{JS_HELPERS}</body></html>"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html)
        
JS_HELPERS = """
<div id="toast"></div>
<script>
function switchTab(groupId, tabId) {
    document.querySelectorAll(`.${groupId}-content`).forEach(el => el.classList.remove('active'));
    document.querySelectorAll(`.${groupId}-btn`).forEach(el => el.classList.remove('active'));
    let tC = document.getElementById(`${groupId}-content-${tabId}`);
    let tB = document.getElementById(`${groupId}-btn-${tabId}`);
    if(tC) tC.classList.add('active');
    if(tB) tB.classList.add('active');
    let url = new URL(window.location); url.searchParams.set(groupId, tabId); window.history.replaceState({}, '', url);
}

function showToast(msg) {
    let x = document.getElementById("toast");
    x.innerText = msg; x.className = "show";
    setTimeout(function(){ x.className = x.className.replace("show", ""); }, 3000);
}

function copyShareLink() {
    let url = window.location.href;
    navigator.clipboard.writeText(url).then(() => showToast("🔗 Share Link Copied to Clipboard! (With active tabs preserved)"));
}

function calcImprovement() {
    let base = document.getElementById("baseline-select");
    if(!base) return;
    base = base.value;
    let tables = document.querySelectorAll(".baseline-table");
    let diffAuc = 0, diffGm = 0, count = 0;
    
    let url = new URL(window.location); url.searchParams.set('baseline', base); window.history.replaceState({}, '', url);

    tables.forEach(t => {
        let rows = t.querySelectorAll("tbody tr");
        let header = Array.from(t.querySelectorAll("th")).map(th=>th.innerText);
        let baseIdx = header.indexOf(base);
        let fplpaIdx = header.indexOf("FPLPA (Ours) 🏆");
        let gainIdx = header.indexOf("FPLPA Gain");
        
        rows.forEach(r => {
            let cells = r.querySelectorAll("td");
            if(cells.length > 5 && baseIdx > -1 && gainIdx > -1) {
                let type = cells[1].innerText;
                let baseVal = parseFloat(cells[baseIdx].innerText);
                let fplpaVal = parseFloat(cells[fplpaIdx].innerText);
                let diff = fplpaVal - baseVal;
                
                let gainCell = cells[gainIdx];
                let sign = diff >= 0 ? "+" : "";
                gainCell.innerHTML = `<span class="${diff >= 0 ? 'gain-pos' : 'gain-neg'}">${sign}${diff.toFixed(4)}</span>`;
                
                if(type === "AUC") diffAuc += diff;
                if(type === "G-mean") diffGm += diff;
                count++;
            }
        });
    });
    
    if(count > 0) {
        let vA = (diffAuc/(count/2)).toFixed(4); let vG = (diffGm/(count/2)).toFixed(4);
        let signA = vA >= 0 ? "+" : ""; let signG = vG >= 0 ? "+" : "";
        document.getElementById("calc-result").innerHTML = `FPLPA outperforms <strong>${base}</strong> by <span class="${vA>=0?'gain-pos':'gain-neg'}">${signA}${vA} AUC</span> and <span class="${vG>=0?'gain-pos':'gain-neg'}">${signG}${vG} G-mean</span> on average!`;
    }
}

function updateRound(sliderVal, tableId) {
    let t = document.getElementById(tableId);
    if(!t) return;
    document.getElementById(`slider-val-${tableId}`).innerText = sliderVal;
    
    let headers = t.querySelectorAll("th");
    for(let i = 2; i < headers.length; i++) {
        headers[i].style.display = (i - 1 <= sliderVal) ? "" : "none";
    }
    t.querySelectorAll("tbody tr").forEach(r => {
        let cells = r.querySelectorAll("td");
        for(let i = 2; i < cells.length; i++) {
            cells[i].style.display = (i - 1 <= sliderVal) ? "" : "none";
        }
    });
    
    if (window[`updateChart_${tableId}`]) {
        window[`updateChart_${tableId}`](sliderVal);
    }
}

window.onload = function() {
    let url = new URL(window.location);
    let tabG = url.searchParams.get("tab-g") || 0;
    let tabB = url.searchParams.get("tab-b") || 0;
    let base = url.searchParams.get("baseline") || "FedAvg";
    
    let btnG = document.getElementById(`tab-g-btn-${tabG}`); if(btnG) btnG.click();
    let btnB = document.getElementById(`tab-b-btn-${tabB}`); if(btnB) btnB.click();
    
    let select = document.getElementById("baseline-select");
    if(select) { select.value = base; calcImprovement(); }
}

function exportCSV(tableId, filename) {
    let t = document.getElementById(tableId);
    if(!t) return;
    let csv = [];
    t.querySelectorAll("tr").forEach(row => {
        let r = [];
        row.querySelectorAll("th, td").forEach(c => {
            if(c.style.display !== "none") r.push(c.innerText.replace(/\\n/g, ' '));
        });
        csv.push(r.join(","));
    });
    let blob = new Blob([csv.join("\\n")], {type: "text/csv"});
    let a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = filename;
    a.click();
    showToast("✅ " + filename + " Exported Successfully!");
}
</script>
"""

def build_eval_matrix_table(sim, table_id=""):
    history_auc = sim["history_auc"]
    history_gmean = sim["history_gmean"]
    active_class = "active" if table_id.endswith("-0") else ""
    html = f"<div class='card tab-content tab-g-content {active_class}' id='tab-g-content-{table_id.split('-')[-1]}'>"
    html += f"<div class='table-responsive'><table id='{table_id}'><thead><tr><th>Project</th><th>Index</th>"
    rounds = sim["rounds"]
    for r in range(1, rounds + 1):
        html += f"<th>R{r}</th>"
    html += "</tr></thead><tbody>"
    
    for cid in history_auc.keys():
        auc_vals = history_auc[cid]
        gm_vals = history_gmean[cid]
        
        peak_auc_rd = int(np.argmax(auc_vals)) + 1
        peak_gm_rd = int(np.argmax(gm_vals)) + 1
        
        html += f"<tr><td>{cid}<br><span class='peak-badge'>🏅 Peak @ R{peak_auc_rd}</span></td><td>AUC</td>"
        for val in auc_vals:
            html += f"<td>{val:.4f}</td>"
        html += "</tr>"
        
        html += f"<tr><td>{cid}<br><span class='peak-badge'>🏅 Peak @ R{peak_gm_rd}</span></td><td>G-mean</td>"
        for val in gm_vals:
            html += f"<td>{val:.4f}</td>"
        html += "</tr>"
        
    html += f"</tbody></table></div>"
    html += f"<div class='slider-container'><label>Rewind to Round: <span id='slider-val-{table_id}'>{rounds}</span></label>"
    html += f"<input type='range' min='1' max='{rounds}' value='{rounds}' class='slider' oninput='updateRound(this.value, \"{table_id}\")'>"
    html += f"</div>"
    
    html += f"<script>window['updateChart_{table_id}'] = function(val) {{"
    html += f"    let aChart = Chart.getChart('auc-chart-{table_id.replace('table-', '')}');"
    html += f"    let gChart = Chart.getChart('gm-chart-{table_id.replace('table-', '')}');"
    html += f"    if(aChart) {{ aChart.data.labels = window['globalLabels'].slice(0, val); aChart.data.datasets.forEach((d, i) => d.data = window['og_data_auc-chart-{table_id.replace('table-', '')}'][i].data.slice(0, val)); aChart.update(); }}"
    html += f"    if(gChart) {{ gChart.data.labels = window['globalLabels'].slice(0, val); gChart.data.datasets.forEach((d, i) => d.data = window['og_data_gm-chart-{table_id.replace('table-', '')}'][i].data.slice(0, val)); gChart.update(); }}"
    html += f"}}</script>"
    
    html += f"<div style='margin-top: 10px;'><button onclick='exportCSV(\"{table_id}\", \"{table_id}.csv\")'>Download CSV ↓</button></div>"
    html += f"<h4 style='color: var(--text-muted); margin-top: 30px;'>Convergence Analysis (Interactive)</h4>"
    return html

def build_baseline_table(sim, table_id=""):
    baselines_list = sim["baselines"]
    clients = list(sim["history_auc"].keys())
    
    active_class = "active" if table_id.endswith("-0") else ""
    html = f"<div class='card tab-content tab-b-content {active_class}' id='tab-b-content-{table_id.split('-')[-1]}'>"
    html += f"<div class='table-responsive'><table id='{table_id}' class='baseline-table'><thead><tr><th>Project</th><th>Index</th>"
    for b in baselines_list:
        html += f"<th>{b}</th>"
    html += "<th class='fplpa-col'>FPLPA (Ours) 🏆</th><th>FPLPA Gain</th></tr></thead><tbody>"
    
    for idx, cid in enumerate(clients):
        b_aucs = sim["collected_b_aucs"][idx]
        b_gmeans = sim["collected_b_gmeans"][idx]
        fplpa_auc = sim["history_auc"][cid][-1]
        fplpa_gmean = sim["history_gmean"][cid][-1]
        
        html += f"<tr><td>{cid}</td><td>AUC</td>"
        for b in baselines_list:
            html += f"<td>{b_aucs[b]:.4f}</td>"
        html += f"<td class='fplpa-col'>{fplpa_auc:.4f}</td><td class='delta-col'></td></tr>"
        
        html += f"<tr><td>{cid}</td><td>G-mean</td>"
        for b in baselines_list:
            html += f"<td>{b_gmeans[b]:.4f}</td>"
        html += f"<td class='fplpa-col'>{fplpa_gmean:.4f}</td><td class='delta-col'></td></tr>"
        
    html += f"</tbody></table></div><div style='margin-top: 10px;'><button onclick='exportCSV(\"{table_id}\", \"{table_id}.csv\")'>Download CSV ↓</button></div>"
    html += f"<h4 style='color: var(--text-muted); margin-top: 30px;'>Average Outcome Comparison</h4>"
    return html

