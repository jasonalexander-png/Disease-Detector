// ---------- Cegah scroll (mouse wheel/touchpad) mengubah nilai input angka ----------
document.querySelectorAll('input[type="number"]').forEach(input => {
  input.addEventListener('wheel', (e) => {
    input.blur();
  }, { passive: true });
});

// ---------- Pill single-select groups (tekanan darah, durasi) ----------
document.querySelectorAll('.pill-group').forEach(group => {
  const hiddenInputId = group.nextElementSibling && group.nextElementSibling.tagName === 'INPUT'
    ? group.nextElementSibling
    : null;

  group.querySelectorAll('.pill').forEach(pill => {
    pill.addEventListener('click', () => {
      const alreadyActive = pill.classList.contains('active');
      group.querySelectorAll('.pill').forEach(p => p.classList.remove('active'));
      if (!alreadyActive) {
        pill.classList.add('active');
        if (hiddenInputId) hiddenInputId.value = pill.dataset.value;
      } else if (hiddenInputId) {
        hiddenInputId.value = '';
      }
    });
  });
});

// ---------- "Tidak tahu / tidak mau menjawab" toggle untuk input angka ----------
document.querySelectorAll('.pill-toggle').forEach(btn => {
  btn.addEventListener('click', () => {
    const targets = btn.dataset.target.split(',').map(id => document.getElementById(id.trim()));
    const nowActive = !btn.classList.contains('active');
    btn.classList.toggle('active', nowActive);
    targets.forEach(input => {
      if (!input) return;
      input.disabled = nowActive;
      if (nowActive) input.value = '';
    });
  });
});

// ---------- Live counter gejala terpilih ----------
const selectedCountEl = document.getElementById('selected-count');

function updateSelectedCount() {
  const n = document.querySelectorAll('input[name="symptoms"]:checked').length;
  selectedCountEl.textContent = `${n} gejala dipilih`;
}

document.querySelectorAll('input[name="symptoms"]').forEach(cb => {
  cb.addEventListener('change', updateSelectedCount);
});

// ---------- Search gejala lintas kategori ----------
const searchInput = document.getElementById('search-symptom');

searchInput.addEventListener('input', () => {
  const q = searchInput.value.toLowerCase().trim();

  document.querySelectorAll('.symptom-category').forEach(category => {
    let anyVisible = false;

    category.querySelectorAll('.symptom-item').forEach(item => {
      const text = item.textContent.toLowerCase();
      const match = q === '' || text.includes(q);
      item.classList.toggle('filtered-out', !match);
      if (match) anyVisible = true;
    });

    category.classList.toggle('filtered-out', !anyVisible);

    if (q !== '' && anyVisible) {
      category.setAttribute('open', '');
    } else if (q === '') {
      category.setAttribute('open', '');
    }
  });
});

// ---------- Submit form ----------
const form = document.getElementById('predict-form');
const resultDiv = document.getElementById('result');

form.addEventListener('submit', async (e) => {
  e.preventDefault();

  const checked = Array.from(
    document.querySelectorAll('input[name="symptoms"]:checked')
  ).map(el => el.value);

  if (checked.length === 0) {
    alert('Pilih minimal 1 gejala dulu ya.');
    return;
  }

  if (checked.length < 3) {
    alert('Pilih minimal 3 gejala dulu ya, supaya hasil prediksi lebih bisa diandalkan.');
    return;
  }

  const payload = {
    symptoms: checked,
    profil: {
      umur: document.getElementById('umur').value,
      berat_badan: document.getElementById('bb').value,
      tinggi_badan: document.getElementById('tb').value,
      tekanan_darah: document.getElementById('tekanan_darah').value,
      durasi_sakit: document.getElementById('durasi_sakit').value
    }
  };

  resultDiv.classList.remove('hidden');
  resultDiv.innerHTML = '<p>Memproses...</p>';
  resultDiv.scrollIntoView({ behavior: 'smooth', block: 'start' });

  try {
    const res = await fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    const data = await res.json();

    if (data.error) {
      resultDiv.innerHTML = `<p class="error">${data.error}</p>`;
      return;
    }

    const info = data.info_tambahan || {};
    const precautionKeys = Object.keys(info).filter(k => k.toLowerCase().includes('precaution'));

    const warningHtml = data.low_confidence ? `
      <div class="confidence-warning">
        ⚠️ <strong>Hasil kurang meyakinkan.</strong> ${data.confidence_message}
      </div>
    ` : '';

    const severity = data.tingkat_keparahan || {};
    const severityClass = {
      'Tinggi': 'severity-high',
      'Sedang': 'severity-medium',
      'Ringan': 'severity-low'
    }[severity.level] || 'severity-low';

    const severityHtml = severity.level ? `
      <div class="severity-badge ${severityClass}">
        Tingkat keparahan gejala: <strong>${severity.level}</strong>
      </div>
    ` : '';

    resultDiv.innerHTML = `
      ${warningHtml}
      ${severityHtml}
      <h2>Kemungkinan: ${data.prediksi_utama}</h2>
      <p><strong>Spesialis disarankan:</strong> ${info.Specialist || '-'}</p>
      <p><strong>Deskripsi:</strong> ${info.Description || '-'}</p>

      <h3>Saran Pencegahan</h3>
      <ul>
        ${precautionKeys.map(k => info[k] ? `<li>${info[k]}</li>` : '').join('') || '<li>-</li>'}
      </ul>

      <h3>Kemungkinan Lain</h3>
      <ul>
        ${data.top3_kemungkinan.map(t => `<li>${t.penyakit} — ${t.probabilitas}%</li>`).join('')}
      </ul>
    `;
  } catch (err) {
    resultDiv.innerHTML = `<p class="error">Terjadi kesalahan koneksi ke server: ${err}</p>`;
  }
});