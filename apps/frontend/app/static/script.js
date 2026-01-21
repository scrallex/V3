
const formatCurrency = (val) => {
    if (val === undefined || val === null) return '--';
    return new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD' }).format(val);
};

const formatNumber = (val, decimals = 2) => {
    if (val === undefined || val === null) return '--';
    return Number(val).toFixed(decimals);
};

async function updateState() {
    try {
        // Parallel fetches
        const [stateResponse, accountResponse, posResponse] = await Promise.all([
            fetch('/api/state').catch(e => ({ json: () => ({}) })),
            fetch('/api/account').catch(e => ({ json: () => ({}) })),
            fetch('/api/positions').catch(e => ({ json: () => ({ positions: [] }) }))
        ]);

        const stateData = await stateResponse.json();
        const accountData = await accountResponse.json();
        const posData = await posResponse.json();

        // Update Account Bar
        if (accountData && accountData.account) {
            const acc = accountData.account;
            document.getElementById('nav-balance').textContent = formatCurrency(acc.balance);
            document.getElementById('nav-val').textContent = formatCurrency(acc.NAV);

            const unrealized = Number(acc.unrealizedPL);
            const pnlEl = document.getElementById('nav-unrealized');
            pnlEl.textContent = formatCurrency(unrealized);
            pnlEl.className = `value ${unrealized >= 0 ? 'text-success' : 'text-danger'}`;

            document.getElementById('nav-margin').textContent = formatCurrency(acc.marginUsed);
            document.getElementById('account-bar').classList.remove('loading');
        }

        // Update Instruments
        for (const [pair, info] of Object.entries(stateData)) {
            const cardId = `card-${pair}`;
            const card = document.getElementById(cardId);
            if (!card) continue;

            card.classList.remove('loading');

            if (info.error) {
                // error state
                continue;
            }

            // Update Signal
            const signalVal = card.querySelector('.signal-value');
            signalVal.textContent = info.signal || 'NEUTRAL';
            signalVal.className = 'value signal-value';
            if (info.signal === 'LONG') signalVal.classList.add('long');
            if (info.signal === 'SHORT') signalVal.classList.add('short');
            if (!info.signal) signalVal.classList.add('neutral');


            // Update Regime
            const regimeVal = card.querySelector('.regime-value');
            regimeVal.textContent = info.regime || 'Unknown';

            const badge = card.querySelector('.badge');
            badge.textContent = info.regime === 'HighVol' ? 'HIGH VOL' : (info.regime === 'LowVol' ? 'LOW VOL' : info.regime);
            badge.className = 'badge';
            if (info.regime === 'HighVol') badge.classList.add('highvol');
            else if (info.regime === 'LowVol') badge.classList.add('lowvol');

            // Update Alignment
            const alignVal = card.querySelector('.alignment-value');
            if (info.ts_ms) {
                const date = new Date(info.ts_ms);
                // Format: HH:MM:SS
                alignVal.textContent = date.toLocaleTimeString('en-US', { hour12: false });
            }

            // Update Hazard Meter
            const meterFill = card.querySelector('.meter-fill');
            const meterText = card.querySelector('.meter-value');

            const pct = info.hazard_pct || 50;
            meterFill.style.width = `${pct}%`;

            // Color based on regime/hazard
            if (info.regime === 'LowVol') {
                meterFill.style.background = '#10b981';
            } else {
                meterFill.style.background = '#f59e0b';
            }

            meterText.textContent = (info.hazard_norm || 0).toFixed(4);
        }

        // Update Positions Table
        const tbody = document.getElementById('positions-body');
        const positions = posData.positions || [];

        if (positions.length === 0) {
            tbody.innerHTML = '<tr class="empty-state"><td colspan="6" style="text-align:center; padding: 2rem; color: var(--text-secondary);">No active positions</td></tr>';
        } else {
            tbody.innerHTML = positions.map(pos => {
                const side = Number(pos.long?.units || 0) > 0 ? 'LONG' : (Number(pos.short?.units || 0) > 0 ? 'SHORT' : 'N/A');
                const units = side === 'LONG' ? pos.long.units : (side === 'SHORT' ? pos.short.units : 0);
                const avgPrice = side === 'LONG' ? pos.long.averagePrice : (side === 'SHORT' ? pos.short.averagePrice : 0);
                const pnl = side === 'LONG' ? pos.long.unrealizedPL : (side === 'SHORT' ? pos.short.unrealizedPL : 0);

                // Note: Current Price isn't directly in position object usually, assume we fetch it or leave blank for now. 
                // We'll just show PnL which is more important.

                return `
                <tr>
                    <td><b>${pos.instrument.replace('_', '/')}</b></td>
                    <td>${Math.abs(units)}</td>
                    <td><span class="badge ${side === 'LONG' ? 'lowvol' : 'highvol'}">${side}</span></td>
                    <td>${formatNumber(avgPrice, 5)}</td>
                    <td>--</td>
                    <td class="${Number(pnl) >= 0 ? 'text-success' : 'text-danger'} font-weight-bold">${formatCurrency(pnl)}</td>
                </tr>
                `;
            }).join('');
        }

    } catch (e) {
        console.error("Failed to fetch state", e);
    }
}

// Poll every 2 seconds
setInterval(updateState, 2000);
updateState();
