// Hypersim (preliminary): a second dataset, shown in its own table below the TartanAir views.
// Reads data/hypersim.json (written by `python -m kpbench.export --hypersim`); nothing here touches the TartanAir state.
(function () {
  const el = document.getElementById('hypersim-body');
  if (!el) return;
  const e = s => String(s).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
  const f = (v, d = 1) => (v == null || !Number.isFinite(v)) ? '–' : v.toFixed(d);
  const sg = (v, d = 1) => (v == null || !Number.isFinite(v)) ? '–' : (v > 0 ? '+' : v < 0 ? '−' : '±') + Math.abs(v).toFixed(d);
  const short = s => s.replace(/^hypersim_/, '');

  fetch('data/hypersim.json', { cache: 'no-cache' })
    .then(r => { if (!r.ok) throw new Error(r.status + ' ' + r.statusText); return r.json(); })
    .then(D => {
      const scenes = D.scenes.map(s => s.id);
      const bins = D.bins.filter(b => b.id !== 'all');
      const head1 = `<tr><th class="l" rowspan="2">run</th><th rowspan="2">fail % (n)</th><th rowspan="2">95 % CI</th>
        <th colspan="${scenes.length}" class="c">fail % per scene</th>
        <th rowspan="2" title="paired block-bootstrap difference on the same queries, 95 % interval">vs sd (points) [95 % CI]</th>
        <th rowspan="2">n_match</th><th rowspan="2">n_inl</th><th rowspan="2">rep1</th><th rowspan="2">rep5</th>
        <th colspan="${bins.length}" class="c">fail % per planarity bin</th></tr>`;
      const nq = Object.fromEntries(D.scenes.map(s => [s.id, s.n_queries]));
      const head2 = `<tr class="sub">${scenes.map(s => `<th>${e(short(s))} (${nq[s]})</th>`).join('')}
        ${bins.map(b => `<th title="${e(b.label)}">${e(b.id.replace('_', ' '))} (${b.n})</th>`).join('')}</tr>`;
      const body = D.runs.map(r => {
        const diff = r.diff ? `${sg(r.diff[0])} <span class="ci">[${sg(r.diff[1])}, ${sg(r.diff[2])}]</span>` : '<span class="muted">baseline</span>';
        return `<tr><td class="l nowrap">${e(r.id)}</td><td><b>${f(r.fail)}</b> <span class="ci">(${r.n})</span></td>
          <td class="ci nowrap">[${f(r.ci[0])}, ${f(r.ci[1])}]</td>
          ${scenes.map(s => `<td>${f(r.scenes[s])}</td>`).join('')}
          <td class="nowrap">${diff}</td><td>${f(r.n_match)}</td><td>${f(r.n_inl)}</td><td>${f(r.rep1, 2)}</td><td>${f(r.rep5, 2)}</td>
          ${bins.map(b => `<td>${f(r.bins[b.id])}</td>`).join('')}</tr>`;
      }).join('');
      el.innerHTML = `
        <div class="tablewrap"><table class="data" id="hypersim-table"><thead>${head1}${head2}</thead><tbody>${body}</tbody></table></div>
        <p class="note">Preliminary: ${scenes.length} scenes and ${D.runs[0].n} queries, so the intervals are wide${
          D.runs.every(r => !r.diff || (r.diff[1] <= 0 && r.diff[2] >= 0)) ? '; no difference to sd is significant (every paired interval contains 0)' : ''}. Fail = no pose, position error &gt; 25 cm or rotation error &gt; 2°. CIs: 1000 bootstrap
          resamples of blocks of ${D.n_boot_block} consecutive frames per scene. Bins use the same plane_frac label and bounds as TartanAir;
          the n per bin is in the header (the ordinary and very flat bins are tiny). Exported ${e(D.generated)}.</p>
        <h3>Protocol</h3>
        <ul class="note hs-proto">${D.protocol.map(p => `<li>${e(p)}</li>`).join('')}</ul>
        <p class="note">Data: ${e(D.source)}. M. Roberts, J. Ramapuram, A. Ranjan, A. Kumar, M. A. Bautista, N. Paczan, R. Webb,
          J. M. Susskind. <i>Hypersim: A Photorealistic Synthetic Dataset for Holistic Indoor Scene Understanding.</i> ICCV 2021.</p>`;
    })
    .catch(err => {
      el.innerHTML = `<p class="bad pad">Could not load data/hypersim.json: ${e(err.message || err)}</p>`;
      console.error(err);
    });
})();
