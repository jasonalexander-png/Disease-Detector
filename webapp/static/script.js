const form = document.getElementById('predict-form');
const resultDiv = document.getElementById('result');
const searchInput = document.getElementById('search-symptom');

searchInput.addEventListener('input', () => {
    const q = searchInput.value.toLowerCase();
    document.querySelectorAll('.symptom-item').forEach(item => {
        const text = item.textContent.toLowerCase();
        item.style.display = text.includes(q) ? 'flex' : 'none';
    });
});

form.addEventListener('submit', async (e) => {
    e.preventDefault();

    const checked = Array.from(
        document.querySelectorAll('input[name="symptoms"]:checked')
    ).map(el => el.value);

    if (checked.length === 0) {
        alert('Pilih minimal 1 gejala dulu ya.');
        return;
    }

    const payload = {
        symptoms: checked,
        profil: {
            umur: document.getElementById('umur').value,
            berat_badan: document.getElementById('bb').value,
            tinggi_badan: document.getElementById('tb').value,
            tekanan_darah: document.getElementById('tekanan_darah').value,
            durasi_sakit: document.getElementById('durasi').value
        }
    };

    resultDiv.classList.remove('hidden');
    resultDiv.innerHTML = '<p>Memproses...</p>';

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

        resultDiv.innerHTML = `
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
