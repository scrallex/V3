
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

            // Update Prob
            const probVal = card.querySelector('.prob-value');
            if (probVal) probVal.textContent = (info.prob || 0).toFixed(4);

            // Update RSI
            const rsiVal = card.querySelector('.rsi-value');
            if (rsiVal) rsiVal.textContent = (info.rsi || 0).toFixed(1);

            // Update Vol
            const volVal = card.querySelector('.vol-value');
            if (volVal) volVal.textContent = (info.volatility || 0).toExponential(2);


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

        // Filter active only: strictly filter out positions where BOTH long and short units are 0
        const activePositions = (posData.positions || []).filter(pos => {
            const longUnits = Number(pos.long?.units || 0);
            const shortUnits = Number(pos.short?.units || 0);
            // Must have non-zero active units
            return Math.abs(longUnits) > 0 || Math.abs(shortUnits) > 0;
        });

        if (activePositions.length === 0) {
            tbody.innerHTML = '<tr class="empty-state"><td colspan="6" style="text-align:center; padding: 2rem; color: var(--text-secondary);">No active positions</td></tr>';
        } else {
            tbody.innerHTML = activePositions.map(pos => {
                const rows = [];
                const longUnits = Number(pos.long?.units || 0);
                const shortUnits = Number(pos.short?.units || 0);
                const pair = pos.instrument;

                // Get Current Price from State Data (Cards)
                // If stateData is missing or price is missing, use '--'
                const priceRaw = stateData[pair] && stateData[pair].price;
                const currentPrice = priceRaw ? formatNumber(priceRaw, 5) : '--';

                if (Math.abs(longUnits) > 0) {
                    const pl = Number(pos.long.unrealizedPL);
                    rows.push(`
                    <tr>
                        <td><b>${pair.replace('_', '/')}</b></td>
                        <td>${Math.abs(longUnits)}</td>
                        <td><span class="badge lowvol">LONG</span></td>
                        <td>${formatNumber(pos.long.averagePrice, 5)}</td>
                        <td>${currentPrice}</td>
                        <td class="${pl >= 0 ? 'text-success' : 'text-danger'} font-weight-bold">${formatCurrency(pl)}</td>
                    </tr>
                    `);
                }

                if (Math.abs(shortUnits) > 0) {
                    const pl = Number(pos.short.unrealizedPL);
                    rows.push(`
                    <tr>
                        <td><b>${pair.replace('_', '/')}</b></td>
                        <td>${Math.abs(shortUnits)}</td>
                        <td><span class="badge highvol">SHORT</span></td>
                        <td>${formatNumber(pos.short.averagePrice, 5)}</td>
                        <td>${currentPrice}</td>
                        <td class="${pl >= 0 ? 'text-success' : 'text-danger'} font-weight-bold">${formatCurrency(pl)}</td>
                    </tr>
                    `);
                }

                return rows.join('');
            }).join('');
        }

    } catch (e) {
        console.error("Failed to fetch state", e);
    }
}

// Version Stamp
console.log("SEP Dashboard v2.1.0 Loaded - Active Position Fixes Applied");

// Poll every 2 seconds
setInterval(updateState, 2000);
updateState();
