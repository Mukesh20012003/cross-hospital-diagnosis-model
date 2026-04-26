// server/static/js/dashboard.js

let accuracyChart = null;
let comparisonChart = null;
let refreshInterval = null;
let lastKnownRound = 0;

// ============ INITIALIZE ============
document.addEventListener('DOMContentLoaded', function() {
    loadDashboard();
    refreshInterval = setInterval(loadDashboard, 10000);
    setInterval(checkNotifications, 5000);
});

// ============ LOAD ALL ============
async function loadDashboard() {
    try {
        await Promise.all([
            loadStats(),
            loadRounds(),
            loadHospitals(),
            loadAuditLogs(),
            loadRoundStatus(),
            loadModelComparison(),
            loadAccuracyTimeline(),
            loadGlobalModels()
        ]);
    } catch (error) {
        console.error('Dashboard load error:', error);
    }
}

// ============ STATS ============
async function loadStats() {
    try {
        const response = await fetch('/api/dashboard_stats');
        const stats = await response.json();
        
        document.getElementById('total-hospitals').textContent = stats.total_hospitals;
        document.getElementById('active-hospitals').textContent = stats.active_hospitals;
        document.getElementById('current-round').textContent = stats.current_round;
        document.getElementById('latest-accuracy').textContent = 
            stats.latest_accuracy ? (stats.latest_accuracy * 100).toFixed(1) + '%' : 'N/A';
        document.getElementById('total-updates').textContent = stats.total_updates;
        document.getElementById('model-downloads').textContent = stats.model_downloads;
    } catch (error) {
        console.error('Stats error:', error);
    }
}

// ============ ROUNDS TABLE WITH IMPROVEMENT ============
async function loadRounds() {
    try {
        const response = await fetch('/rounds');
        const rounds = await response.json();
        
        const tbody = document.getElementById('rounds-table-body');
        if (tbody) {
            tbody.innerHTML = '';
            let prevAccuracy = null;
            
            // Reverse to get chronological order for improvement calc
            const chronological = [...rounds].reverse();
            const improvements = {};
            chronological.forEach(round => {
                if (round.global_accuracy && prevAccuracy) {
                    improvements[round.round_number] = ((round.global_accuracy - prevAccuracy) * 100).toFixed(2);
                }
                if (round.global_accuracy) prevAccuracy = round.global_accuracy;
            });
            
            rounds.forEach(round => {
                const imp = improvements[round.round_number];
                const impDisplay = imp ? 
                    `<span style="color: ${parseFloat(imp) >= 0 ? '#2e7d32' : '#c62828'}; font-weight:600;">
                        ${parseFloat(imp) >= 0 ? '+' : ''}${imp}%
                    </span>` : '-';
                
                const row = document.createElement('tr');
                row.innerHTML = `
                    <td><strong>Round ${round.round_number}</strong></td>
                    <td><span class="badge ${round.status}">${round.status}</span></td>
                    <td>${round.num_participants} / ${round.target_participants}</td>
                    <td>${round.global_accuracy ? (round.global_accuracy * 100).toFixed(2) + '%' : '-'}</td>
                    <td>${impDisplay}</td>
                    <td>${round.global_loss ? round.global_loss.toFixed(4) : '-'}</td>
                    <td>${formatDate(round.started_at)}</td>
                `;
                tbody.appendChild(row);
            });
        }
        
        updateAccuracyChart(rounds.reverse());
    } catch (error) {
        console.error('Rounds error:', error);
    }
}

// ============ HOSPITALS ============
async function loadHospitals() {
    try {
        const response = await fetch('/hospitals');
        const hospitals = await response.json();
        
        const tbody = document.getElementById('hospitals-table-body');
        if (tbody) {
            tbody.innerHTML = '';
            hospitals.forEach(hospital => {
                const row = document.createElement('tr');
                row.innerHTML = `
                    <td><strong>${hospital.name}</strong></td>
                    <td>${hospital.location || '-'}</td>
                    <td>${hospital.data_size} samples</td>
                    <td><span class="badge ${hospital.is_active ? 'active' : 'failed'}">${hospital.is_active ? 'Active' : 'Inactive'}</span></td>
                    <td>${formatDate(hospital.last_seen)}</td>
                `;
                tbody.appendChild(row);
            });
        }
    } catch (error) {
        console.error('Hospitals error:', error);
    }
}

// ============ GLOBAL MODELS TABLE ============
async function loadGlobalModels() {
    try {
        const response = await fetch('/global_models');
        const models = await response.json();
        
        const tbody = document.getElementById('models-table-body');
        if (tbody) {
            tbody.innerHTML = '';
            models.forEach(model => {
                const row = document.createElement('tr');
                row.innerHTML = `
                    <td><strong>${model.version}</strong></td>
                    <td>Round ${model.round_id}</td>
                    <td>${model.accuracy ? (model.accuracy * 100).toFixed(2) + '%' : '-'}</td>
                    <td>${model.loss ? model.loss.toFixed(4) : '-'}</td>
                    <td>${model.download_count}</td>
                    <td>${formatDate(model.created_at)}</td>
                `;
                tbody.appendChild(row);
            });
            if (models.length === 0) {
                tbody.innerHTML = '<tr><td colspan="6">No global models yet. Run training rounds first.</td></tr>';
            }
        }
    } catch (error) {
        console.error('Models error:', error);
    }
}

// ============ AUDIT LOGS ============
async function loadAuditLogs() {
    try {
        const response = await fetch('/audit_logs?limit=15');
        const logs = await response.json();
        
        const container = document.getElementById('audit-logs-container');
        if (container) {
            container.innerHTML = '';
            const iconMap = {
                'register_hospital': '🏥', 'start_round': '🚀',
                'submit_update': '📤', 'aggregation_complete': '✅',
                'download_model': '📥', 'upload_metadata': '📋',
                'export_report': '📄', 'update_validation_warning': '⚠️'
            };
            
            logs.forEach(log => {
                const div = document.createElement('div');
                div.className = 'audit-item';
                div.innerHTML = `
                    <span class="audit-icon">${iconMap[log.action] || '📋'}</span>
                    <div class="audit-content">
                        <div class="action">${formatAction(log.action)}</div>
                        <div class="details">${log.details || ''}</div>
                    </div>
                    <span class="audit-time">${formatDate(log.timestamp)}</span>
                `;
                container.appendChild(div);
            });
        }
    } catch (error) {
        console.error('Audit logs error:', error);
    }
}

// ============ ROUND STATUS ============
async function loadRoundStatus() {
    try {
        const response = await fetch('/round_status');
        const status = await response.json();
        
        const banner = document.getElementById('round-status-banner');
        if (banner) {
            if (status.status === 'in_progress') {
                const progress = (status.participants / status.target) * 100;
                banner.innerHTML = `
                    <div class="round-info">
                        <h3>🔄 Round ${status.round_number} In Progress</h3>
                        <p>${status.participants} of ${status.target} hospitals have submitted (${status.remaining} remaining)</p>
                        <div class="progress-bar"><div class="progress" style="width: ${progress}%"></div></div>
                    </div>
                    <div class="btn-group">
                        <button class="btn btn-primary" onclick="loadDashboard()">🔄 Refresh</button>
                    </div>
                `;
            } else if (status.status === 'completed') {
                let improvementText = '';
                if (status.previous_accuracy && status.global_accuracy) {
                    const diff = ((status.global_accuracy - status.previous_accuracy) * 100).toFixed(1);
                    improvementText = ` | Improved from ${(status.previous_accuracy*100).toFixed(1)}% → ${(status.global_accuracy*100).toFixed(1)}% (${parseFloat(diff) >= 0 ? '+' : ''}${diff}%)`;
                }
                
                banner.innerHTML = `
                    <div class="round-info">
                        <h3>✅ Round ${status.round_number} Completed — ${status.participants} hospitals contributed — Global accuracy: ${status.global_accuracy ? (status.global_accuracy*100).toFixed(1) + '%' : 'N/A'}${improvementText}</h3>
                        <p>All participating hospitals have been notified of the new global model.</p>
                    </div>
                    <div class="btn-group">
                        <button class="btn btn-success" onclick="startNewRound()">🚀 Start New Round</button>
                    </div>
                `;
            } else {
                banner.innerHTML = `
                    <div class="round-info">
                        <h3>📋 No Active Rounds</h3>
                        <p>Start a new round to begin federated training</p>
                    </div>
                    <div class="btn-group">
                        <button class="btn btn-success" onclick="startNewRound()">🚀 Start First Round</button>
                    </div>
                `;
            }
        }
    } catch (error) {
        console.error('Round status error:', error);
    }
}

// ============ ACCURACY TIMELINE (Improvement Summary) ============
async function loadAccuracyTimeline() {
    try {
        const response = await fetch('/api/accuracy_timeline');
        const data = await response.json();
        
        const container = document.getElementById('improvement-summary');
        if (container && data.summary) {
            const s = data.summary;
            if (s.first_round_accuracy && s.latest_accuracy) {
                container.innerHTML = `
                    <div style="display: flex; justify-content: space-around; align-items: center; flex-wrap: wrap; gap: 20px;">
                        <div style="text-align: center;">
                            <div style="font-size: 0.85em; color: #888; text-transform: uppercase;">First Round</div>
                            <div style="font-size: 2em; font-weight: 700; color: #ef6c00;">${s.first_round_accuracy}</div>
                        </div>
                        <div style="font-size: 2em; color: #1a237e;">→</div>
                        <div style="text-align: center;">
                            <div style="font-size: 0.85em; color: #888; text-transform: uppercase;">Latest Round</div>
                            <div style="font-size: 2em; font-weight: 700; color: #2e7d32;">${s.latest_accuracy}</div>
                        </div>
                        <div style="text-align: center; background: #e8f5e9; padding: 15px 25px; border-radius: 10px;">
                            <div style="font-size: 0.85em; color: #888; text-transform: uppercase;">Total Improvement</div>
                            <div style="font-size: 2em; font-weight: 700; color: #2e7d32;">${s.total_improvement || 'N/A'}</div>
                        </div>
                        <div style="text-align: center;">
                            <div style="font-size: 0.85em; color: #888; text-transform: uppercase;">Rounds Completed</div>
                            <div style="font-size: 2em; font-weight: 700; color: #1a237e;">${s.total_rounds}</div>
                        </div>
                    </div>
                `;
            } else {
                container.innerHTML = '<p style="color:#888;">Complete training rounds to see improvement data.</p>';
            }
        }
    } catch (error) {
        console.error('Timeline error:', error);
    }
}

// ============ MODEL COMPARISON CHART ============
async function loadModelComparison() {
    try {
        const response = await fetch('/api/model_comparison');
        const data = await response.json();
        
        const ctx = document.getElementById('comparison-chart');
        if (!ctx || !data.comparison.length) return;
        
        const labels = data.comparison.map(c => `Round ${c.round_number}`);
        const globalAcc = data.comparison.map(c => c.global_accuracy ? (c.global_accuracy * 100).toFixed(2) : 0);
        const bestSingle = data.comparison.map(c => (c.best_single_hospital * 100).toFixed(2));
        const avgSingle = data.comparison.map(c => (c.avg_single_hospital * 100).toFixed(2));
        
        if (comparisonChart) comparisonChart.destroy();
        
        comparisonChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [
                    {
                        label: 'Global Model (Federated)',
                        data: globalAcc,
                        backgroundColor: 'rgba(26, 35, 126, 0.8)',
                        borderColor: '#1a237e',
                        borderWidth: 1
                    },
                    {
                        label: 'Best Single Hospital',
                        data: bestSingle,
                        backgroundColor: 'rgba(239, 108, 0, 0.6)',
                        borderColor: '#ef6c00',
                        borderWidth: 1
                    },
                    {
                        label: 'Average Single Hospital',
                        data: avgSingle,
                        backgroundColor: 'rgba(158, 158, 158, 0.5)',
                        borderColor: '#9e9e9e',
                        borderWidth: 1
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { position: 'top' } },
                scales: {
                    y: { beginAtZero: false, title: { display: true, text: 'Accuracy (%)' } }
                }
            }
        });
    } catch (error) {
        console.error('Comparison error:', error);
    }
}

// ============ ACCURACY CHART ============
function updateAccuracyChart(rounds) {
    const ctx = document.getElementById('accuracy-chart');
    if (!ctx) return;
    
    const completedRounds = rounds.filter(r => r.status === 'completed' && r.global_accuracy);
    const labels = completedRounds.map(r => `Round ${r.round_number}`);
    const accuracies = completedRounds.map(r => (r.global_accuracy * 100).toFixed(2));
    const losses = completedRounds.map(r => r.global_loss ? r.global_loss.toFixed(4) : 0);
    
    if (accuracyChart) accuracyChart.destroy();
    
    accuracyChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'Global Accuracy (%)',
                    data: accuracies,
                    borderColor: '#1a237e',
                    backgroundColor: 'rgba(26, 35, 126, 0.1)',
                    borderWidth: 3, fill: true, tension: 0.3,
                    pointRadius: 6, pointBackgroundColor: '#1a237e'
                },
                {
                    label: 'Global Loss',
                    data: losses,
                    borderColor: '#ef6c00',
                    backgroundColor: 'rgba(239, 108, 0, 0.1)',
                    borderWidth: 2, fill: false, tension: 0.3,
                    pointRadius: 4, pointBackgroundColor: '#ef6c00',
                    yAxisID: 'y1'
                }
            ]
        },
        options: {
            responsive: true, maintainAspectRatio: false,
            plugins: { legend: { position: 'top' } },
            scales: {
                y: { beginAtZero: false, title: { display: true, text: 'Accuracy (%)' } },
                y1: { position: 'right', beginAtZero: true, title: { display: true, text: 'Loss' }, grid: { drawOnChartArea: false } }
            }
        }
    });
}

// ============ NOTIFICATIONS ============
async function checkNotifications() {
    try {
        const response = await fetch(`/notifications?since_round=${lastKnownRound}`);
        const data = await response.json();
        
        if (data.count > 0) {
            data.notifications.forEach(n => {
                showNotification(n.message, 'success');
            });
            lastKnownRound = data.latest_round;
            loadDashboard();
        }
    } catch (error) {}
}

// ============ START ROUND ============
async function startNewRound() {
    try {
        const response = await fetch('/start_round', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ target_participants: 3 })
        });
        
        if (response.ok) {
            const result = await response.json();
            showNotification(`Round ${result.round_number} started! Waiting for hospital submissions...`, 'success');
            loadDashboard();
        } else {
            const error = await response.json();
            showNotification(error.detail, 'error');
        }
    } catch (error) {
        showNotification('Failed to start round', 'error');
    }
}

// ============ UPLOAD METADATA ============
async function uploadMetadata(event) {
    event.preventDefault();
    
    const formData = new FormData();
    formData.append('api_key', document.getElementById('meta-api-key').value);
    formData.append('hospital_name', document.getElementById('meta-hospital-name').value);
    formData.append('data_description', document.getElementById('meta-description').value);
    formData.append('num_samples', document.getElementById('meta-samples').value);
    formData.append('data_type', document.getElementById('meta-data-type').value);
    
    try {
        const response = await fetch('/upload_metadata', {
            method: 'POST',
            body: formData
        });
        
        const resultDiv = document.getElementById('metadata-result');
        if (response.ok) {
            const result = await response.json();
            resultDiv.innerHTML = `<div style="color: #2e7d32; font-weight: 600;">✅ ${result.message}</div>`;
            showNotification('Metadata uploaded successfully!', 'success');
            loadDashboard();
        } else {
            const error = await response.json();
            resultDiv.innerHTML = `<div style="color: #c62828; font-weight: 600;">❌ ${error.detail}</div>`;
        }
    } catch (error) {
        showNotification('Upload failed', 'error');
    }
}

// ============ EXPORT FUNCTIONS ============
async function exportCSV() {
    window.open('/export/csv', '_blank');
    showNotification('CSV report downloaded!', 'success');
}

async function exportPDF() {
    showNotification('Generating PDF report...', 'info');
    window.open('/export/pdf', '_blank');
}

async function exportAuditPDF() {
    window.open('/export/audit_csv', '_blank');
    showNotification('Audit log exported!', 'success');
}

// ============ HELPERS ============
function formatDate(dateStr) {
    if (!dateStr) return '-';
    const date = new Date(dateStr);
    return date.toLocaleDateString('en-US', { 
        month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' 
    });
}

function formatAction(action) {
    const map = {
        'register_hospital': 'Hospital Registered',
        'start_round': 'Round Started',
        'submit_update': 'Update Submitted',
        'aggregation_complete': 'Aggregation Complete',
        'download_model': 'Model Downloaded',
        'upload_metadata': 'Metadata Uploaded',
        'export_report': 'Report Generated',
        'update_validation_warning': 'Validation Warning'
    };
    return map[action] || action;
}

function showNotification(message, type = 'info') {
    const notif = document.createElement('div');
    notif.className = `notification ${type}`;
    notif.textContent = message;
    document.body.appendChild(notif);
    setTimeout(() => {
        notif.style.animation = 'slideIn 0.3s ease reverse';
        setTimeout(() => notif.remove(), 300);
    }, 4000);
}